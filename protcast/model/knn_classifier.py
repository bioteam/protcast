"""knn_classifier.py

K-Nearest Neighbors classifier for protein function prediction using
ESM embeddings. Designed as a direct comparison to MultiLabelClassifier:
same data structures, same metrics (Fmax, Smin), same MLflow schema.

Core idea: build a single NearestNeighbors index over all training protein
embeddings. For a query protein, find K neighbors and aggregate their
multi-hot GO labels with distance-weighted voting to produce per-term
scores in [0, 1] — directly compatible with Fmax threshold sweep.
"""

from __future__ import annotations

import os
import time
import pickle
import joblib
import numpy as np
from pathlib import Path
from typeguard import typechecked
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from protcast.model.stats.utils import calculate_fmax, calculate_smin
from protcast.model.multilabel_classifier import GOEncoder, get_confidence_label


@typechecked
class KNNClassifier:
    """K-Nearest Neighbors multi-label classifier for protein GO prediction.

    Uses a single NearestNeighbors index over ESM protein embeddings.
    For each query protein, finds K nearest training proteins and aggregates
    their GO annotations via distance-weighted voting.

    Produces the same metrics (Fmax, Smin) and logs to the same MLflow
    schema as MultiLabelClassifier for head-to-head comparison.

    Parameters
    ----------
    verbose : bool
        Whether to print progress information.
    protein_embeddings : dict
        {protein_id: np.ndarray} mapping protein IDs to embedding vectors.
    protein_go_terms : dict
        {protein_id: set[str]} mapping protein IDs to their GO term annotations.
    go_ids : list
        Ordered list of GO term IDs to predict (defines label columns).
    config : dict
        Configuration dictionary from config.json.
    id : str
        Identifier for this model run (used in filenames).
    use_mlflow : bool
        Whether to log to MLflow.
    go_dag : object or None
        GO DAG for depth-level metric breakdowns. Optional.
    """

    def __init__(
        self,
        verbose: bool,
        protein_embeddings: dict,
        protein_go_terms: dict,
        go_ids: list,
        config: dict,
        id: str,
        use_mlflow: bool = False,
        go_dag: object = None,
    ) -> None:
        self.verbose = verbose
        self.protein_embeddings = protein_embeddings
        self.protein_go_terms = protein_go_terms
        self.go_ids = sorted(go_ids)
        self.use_mlflow = use_mlflow
        self.id = id
        self.go_dag = go_dag

        # Set instance attributes from config
        self.params = config
        for key, value in config.items():
            setattr(self, key.lower(), value)

        # KNN-specific defaults (overridden by config if present)
        if not hasattr(self, "knn_n_neighbors"):
            self.knn_n_neighbors = 10
        if not hasattr(self, "knn_metric"):
            self.knn_metric = "cosine"
        if not hasattr(self, "knn_weights"):
            self.knn_weights = "distance"
        if not hasattr(self, "knn_algorithm"):
            self.knn_algorithm = "auto"

        self.training_time = 0
        self.logging_time = 0

        self._mlflow = None
        if self.use_mlflow:
            from protcast.utils.mlflow_utils import init_mlflow

            self._mlflow = init_mlflow(
                experiment_name=config.get(
                    "EXPERIMENT_NAME", "Default Experiment"
                ),
                repo_owner=config.get("DAGSHUB_REPO_OWNER", "aakpan"),
                repo_name=config.get("DAGSHUB_REPO_NAME", "my-first-repo"),
                verbose=self.verbose,
            )

    @typechecked
    def run(self) -> None:
        """Main training + evaluation orchestration."""
        self.start_time = time.time()

        if self.use_mlflow and self._mlflow is not None:
            self._mlflow.start_run()

        self.prepare_data()
        self.build_model()
        self.train_model()
        if self.use_mlflow:
            self.log_model()

    @typechecked
    def prepare_data(self) -> None:
        """Build feature matrix X and multi-hot label matrix y.

        Identical to MultiLabelClassifier.prepare_data() — same split,
        same random_state, so both models see exactly the same data.
        """
        go_encoder = GOEncoder(self.id)
        go_encoder.fit(self.go_ids)
        go_encoder.save()
        self.go_encoder = go_encoder

        protein_ids = sorted(
            set(self.protein_embeddings.keys()) & set(self.protein_go_terms.keys())
        )

        if not protein_ids:
            raise ValueError(
                "No proteins found with both embeddings and GO annotations."
            )

        embedding_dim = len(next(iter(self.protein_embeddings.values())))
        self.vector_length = embedding_dim
        X_list = []
        y_list = []

        for pid in protein_ids:
            embedding = self.protein_embeddings[pid]
            if hasattr(embedding, "astype"):
                X_list.append(embedding.astype(np.float32))
            else:
                X_list.append(np.array(embedding, dtype=np.float32))

            label = np.zeros(len(self.go_ids), dtype=np.float32)
            for go_id in self.protein_go_terms[pid]:
                if go_id in go_encoder.go_to_int:
                    label[go_encoder.go_to_int[go_id]] = 1.0
            y_list.append(label)

        self.X = np.vstack(X_list)
        self.y = np.array(y_list)
        self.protein_ids = protein_ids

        if self.verbose:
            num_annotations = int(self.y.sum())
            avg_labels = num_annotations / len(protein_ids)
            print(f"Proteins: {len(protein_ids)}")
            print(f"GO terms: {len(self.go_ids)}")
            print(f"Total annotations: {num_annotations}")
            print(f"Avg GO terms per protein: {avg_labels:.1f}")
            print(f"Embedding dim: {embedding_dim}")
            print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")

    @typechecked
    def build_model(self) -> None:
        """Build the NearestNeighbors index."""
        # Clamp K to not exceed the training set size
        validation_split = getattr(self, "validation_split", 0.2)
        max_k = int(self.X.shape[0] * (1 - validation_split)) - 1
        k = min(self.knn_n_neighbors, max(1, max_k))
        if k != self.knn_n_neighbors and self.verbose:
            print(
                f"Clamped K from {self.knn_n_neighbors} to {k} "
                f"(training set too small)"
            )
        self.knn_n_neighbors = k

        self.nn = NearestNeighbors(
            n_neighbors=self.knn_n_neighbors,
            metric=self.knn_metric,
            algorithm=self.knn_algorithm,
        )

        if self.verbose:
            print(f"KNN config: K={self.knn_n_neighbors}, "
                  f"metric={self.knn_metric}, "
                  f"weights={self.knn_weights}, "
                  f"algorithm={self.knn_algorithm}")

    def train_model(self) -> None:
        """Fit the KNN index on training data and evaluate on validation.

        Uses the same train/test split as MultiLabelClassifier (random_state=42)
        for a fair comparison.
        """
        if self.verbose:
            print("Fitting KNN index...")

        validation_split = getattr(self, "validation_split", 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y,
            test_size=validation_split,
            random_state=42,
        )

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

        train_start = time.time()

        # Fit the index
        self.nn.fit(X_train)

        # Predict on validation set
        y_val_pred = self.predict(X_val)

        self.training_time = time.time() - train_start

        # Compute CAFA metrics
        fmax, fmax_threshold = calculate_fmax(y_val, y_val_pred)
        smin, smin_threshold = calculate_smin(y_val, y_val_pred)

        self.best_fmax = fmax
        self.best_threshold = fmax_threshold
        self.best_smin = smin
        self.smin_threshold = smin_threshold
        self.y_val_pred = y_val_pred

        if self.verbose:
            print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
            print(f"KNN fitting + eval completed in {self.training_time:.2f}s")
            print(f"Best Fmax: {fmax:.4f} (threshold={fmax_threshold:.2f})")
            print(f"Best Smin: {smin:.4f} (threshold={smin_threshold:.2f})")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict multi-label scores for a set of protein embeddings.

        For each query protein:
        1. Find K nearest neighbors in the training set
        2. Weight by inverse distance (or uniform)
        3. Compute weighted average of neighbor labels per GO term

        Parameters
        ----------
        X : np.ndarray
            Embedding matrix, shape (n_samples, embed_dim).

        Returns
        -------
        np.ndarray
            Score matrix, shape (n_samples, n_classes), values in [0, 1].
        """
        distances, indices = self.nn.kneighbors(X)
        n_samples = X.shape[0]
        n_classes = self.y_train.shape[1]
        scores = np.zeros((n_samples, n_classes), dtype=np.float32)

        for i in range(n_samples):
            if self.knn_weights == "distance":
                weights = 1.0 / (distances[i] + 1e-8)
            else:
                weights = np.ones_like(distances[i])

            neighbor_labels = self.y_train[indices[i]]  # (K, n_classes)
            scores[i] = np.average(neighbor_labels, axis=0, weights=weights)

        return scores

    def compute_depth_metrics(self, y_true, y_pred):
        """Compute Fmax broken down by GO term depth in the DAG.

        Groups GO terms by their depth (longest path to root) and computes
        Fmax for each depth level. This is the key analysis for comparing
        KNN vs neural network on rare/specific terms.

        Returns
        -------
        dict
            {depth: {"fmax": float, "threshold": float, "n_terms": int,
                      "avg_train_count": float}}
        """
        if self.go_dag is None:
            return {}

        # Map each GO term index to its depth
        term_depths = {}
        for i, go_id in enumerate(self.go_ids):
            if go_id in self.go_dag.go_terms_map:
                term = self.go_dag.go_terms_map[go_id]
                term_depths[i] = term.depth

        # Group term indices by depth
        from collections import defaultdict
        depth_groups = defaultdict(list)
        for idx, depth in term_depths.items():
            depth_groups[depth].append(idx)

        # Compute Fmax per depth level
        results = {}
        train_counts = self.y_train.sum(axis=0) if hasattr(self, "y_train") else self.y.sum(axis=0)

        for depth, term_indices in sorted(depth_groups.items()):
            # Slice y_true and y_pred to only these GO terms
            y_true_subset = y_true[:, term_indices]
            y_pred_subset = y_pred[:, term_indices]

            # Only evaluate if there are actual annotations
            if y_true_subset.sum() == 0:
                continue

            fmax, threshold = calculate_fmax(y_true_subset, y_pred_subset)
            avg_count = float(train_counts[term_indices].mean())

            results[depth] = {
                "fmax": fmax,
                "threshold": threshold,
                "n_terms": len(term_indices),
                "avg_train_count": avg_count,
            }

        return results

    def compute_frequency_metrics(self, y_true, y_pred):
        """Compute Fmax broken down by training set frequency.

        Buckets: rare (<50), medium (50-500), common (>500).

        Returns
        -------
        dict
            {bucket_name: {"fmax": float, "threshold": float, "n_terms": int,
                            "avg_train_count": float}}
        """
        train_counts = self.y_train.sum(axis=0) if hasattr(self, "y_train") else self.y.sum(axis=0)

        buckets = {
            "rare_lt50": [],
            "medium_50_500": [],
            "common_gt500": [],
        }

        for i in range(len(self.go_ids)):
            count = train_counts[i]
            if count < 50:
                buckets["rare_lt50"].append(i)
            elif count <= 500:
                buckets["medium_50_500"].append(i)
            else:
                buckets["common_gt500"].append(i)

        results = {}
        for bucket_name, term_indices in buckets.items():
            if not term_indices:
                continue

            y_true_subset = y_true[:, term_indices]
            y_pred_subset = y_pred[:, term_indices]

            if y_true_subset.sum() == 0:
                continue

            fmax, threshold = calculate_fmax(y_true_subset, y_pred_subset)
            avg_count = float(train_counts[term_indices].mean())

            results[bucket_name] = {
                "fmax": fmax,
                "threshold": threshold,
                "n_terms": len(term_indices),
                "avg_train_count": avg_count,
            }

        return results

    def save_model(self, path: str = None) -> str:
        """Save the trained KNN model to disk.

        Serializes the NearestNeighbors index, training labels, encoder,
        threshold, and hyperparameters into a single joblib file.

        Returns the path to the saved file.
        """
        if path is None:
            path = f"{self.get_name()}.joblib"

        artifact = {
            "nn": self.nn,
            "y_train": self.y_train,
            "go_ids": self.go_ids,
            "best_threshold": self.best_threshold,
            "best_fmax": self.best_fmax,
            "knn_n_neighbors": self.knn_n_neighbors,
            "knn_metric": self.knn_metric,
            "knn_weights": self.knn_weights,
            "vector_length": self.vector_length,
        }

        joblib.dump(artifact, path)
        if self.verbose:
            print(f"Model saved to {path}")
        return path

    @classmethod
    def load_model(cls, path: str) -> dict:
        """Load a trained KNN model from disk.

        Returns the artifact dict with keys: nn, y_train, go_ids,
        best_threshold, best_fmax, knn_n_neighbors, knn_metric,
        knn_weights, vector_length.
        """
        return joblib.load(path)

    @classmethod
    def predict_from_artifact(cls, artifact: dict, X: np.ndarray) -> np.ndarray:
        """Predict using a loaded model artifact.

        Parameters
        ----------
        artifact : dict
            Loaded via KNNClassifier.load_model().
        X : np.ndarray
            Query embeddings, shape (n_samples, embed_dim).

        Returns
        -------
        np.ndarray
            Score matrix, shape (n_samples, n_classes).
        """
        distances, indices = artifact["nn"].kneighbors(X)
        y_train = artifact["y_train"]
        weights_mode = artifact.get("knn_weights", "distance")
        n_samples = X.shape[0]
        n_classes = y_train.shape[1]
        scores = np.zeros((n_samples, n_classes), dtype=np.float32)

        for i in range(n_samples):
            if weights_mode == "distance":
                weights = 1.0 / (distances[i] + 1e-8)
            else:
                weights = np.ones_like(distances[i])

            neighbor_labels = y_train[indices[i]]
            scores[i] = np.average(neighbor_labels, axis=0, weights=weights)

        return scores

    @typechecked
    def log_model(self) -> None:
        """Log model, metrics, and artifacts to MLflow.

        Same schema as MultiLabelClassifier.log_model() so runs are
        directly comparable in the MLflow UI.
        """
        log_start_time = time.time()
        print("\n--- Starting MLflow Logging ---")

        mlflow = self._mlflow
        if mlflow is None:
            if self.verbose:
                print("mlflow not available; skipping logging")
            return

        try:
            from protcast.utils.mlflow_utils import save_run_metadata
        except Exception as e:
            if self.verbose:
                print("mlflow sub-imports failed; skipping logging:", e)
            return

        # Log parameters
        print("  > Logging parameters...", end="", flush=True)
        mlflow.log_params(self.params)
        mlflow.log_param("model_type", "knn")
        mlflow.log_param("input_source", "esm_embeddings")
        mlflow.log_param("num_classes", len(self.go_ids))
        mlflow.log_param("feature_vector_length", self.vector_length)
        mlflow.log_param("best_threshold", round(self.best_threshold, 3))
        mlflow.log_param("knn_n_neighbors", self.knn_n_neighbors)
        mlflow.log_param("knn_metric", self.knn_metric)
        mlflow.log_param("knn_weights", self.knn_weights)
        print(" done.")

        # Log dataset metadata
        print("  > Logging dataset metadata...", end="", flush=True)
        total_samples = self.X.shape[0]
        train_samples = self.X_train.shape[0]
        val_samples = self.X_val.shape[0]
        mlflow.log_metric("total_samples", total_samples)
        mlflow.log_metric("train_samples", train_samples)
        mlflow.log_metric("val_samples", val_samples)
        mlflow.log_metric("avg_labels_per_protein", float(self.y.sum() / self.y.shape[0]))
        print(" done.")

        # Log per-class sample counts
        print("  > Logging per-class sample counts...", end="", flush=True)
        for i, go_id in enumerate(self.go_ids):
            count = int(self.y[:, i].sum())
            mlflow.log_metric(f"samples_{go_id}", count)
        print(" done.")

        # Log CAFA metrics
        print("  > Logging CAFA metrics...", end="", flush=True)
        mlflow.log_metric("val_fmax", round(self.best_fmax, 4))
        mlflow.log_metric("best_threshold", round(self.best_threshold, 4))
        mlflow.log_metric("val_smin", round(self.best_smin, 4))
        mlflow.log_metric("smin_threshold", round(self.smin_threshold, 4))
        print(" done.")

        # Log depth-level metrics
        depth_metrics = self.compute_depth_metrics(self.y_val, self.y_val_pred)
        if depth_metrics:
            print("  > Logging depth-level metrics...", end="", flush=True)
            for depth, metrics in depth_metrics.items():
                mlflow.log_metric(f"fmax_depth_{depth}", round(metrics["fmax"], 4))
                mlflow.log_metric(f"n_terms_depth_{depth}", metrics["n_terms"])
                mlflow.log_metric(f"avg_count_depth_{depth}", round(metrics["avg_train_count"], 1))
            print(" done.")

        # Log frequency-bucket metrics
        freq_metrics = self.compute_frequency_metrics(self.y_val, self.y_val_pred)
        if freq_metrics:
            print("  > Logging frequency-bucket metrics...", end="", flush=True)
            for bucket, metrics in freq_metrics.items():
                mlflow.log_metric(f"fmax_{bucket}", round(metrics["fmax"], 4))
                mlflow.log_metric(f"n_terms_{bucket}", metrics["n_terms"])
            print(" done.")

        # Log GO term list
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="go_terms_"
        ) as f:
            for go_id in self.go_ids:
                f.write(f"{go_id}\n")
            go_terms_path = f.name
        mlflow.log_artifact(go_terms_path, artifact_path="metadata")
        os.remove(go_terms_path)

        # Save and log the model artifact
        print("  > Saving model artifact...", end="", flush=True)
        model_path = self.save_model()
        mlflow.log_artifact(model_path, artifact_path="model")
        print(" done.")

        mlflow.set_tag("Training Info", "KNNClassifier full logging")
        mlflow.set_tag("model_type", "knn")

        self.logging_time = time.time() - log_start_time
        mlflow.log_metric("training_time_seconds", round(self.training_time, 2))
        mlflow.log_metric("total_logging_time_seconds", round(self.logging_time, 2))

        run_id = mlflow.active_run().info.run_id
        save_run_metadata(
            model_name=self.get_name(),
            run_id=run_id,
            experiment_name=self.params.get(
                "EXPERIMENT_NAME", "Default Experiment"
            ),
        )

        mlflow.end_run()
        print("--- MLflow Logging Complete ---")

    @typechecked
    def get_name(self) -> str:
        """Generate model name using id."""
        return f"{self.id}_knn"

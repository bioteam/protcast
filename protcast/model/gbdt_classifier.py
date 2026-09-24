"""gbdt_classifier.py

Gradient-boosted decision tree (LightGBM) classifier for protein function
prediction. Designed as a third, tree-based arm alongside KNNClassifier and
MultiLabelClassifier: same data structures, same train/val split, same metrics
(Fmax, Smin), same MLflow schema.

Why a tree arm
--------------
KNN is distance-based and the MLP is representation-learning; a boosted tree
ensemble is an unrelated inductive bias. If concatenating classical feature
vectors (PseKRAAC) to ESM-C embeddings helps under all three priors, the gain is
attributable to the features, not to one architecture's quirks.

Trees are scale-invariant, so this classifier consumes RAW features — no
StandardScaler. That removes the block-scaling confound that the neural arms
must manage for multi-block pooling (mean_max_std) and for ESM ⊕ FV inputs.

Multi-label handling
--------------------
y is a (n_proteins, n_go_terms) multi-hot matrix. One binary LightGBM booster is
trained per GO term (one-vs-rest). The design matrix is binned ONCE into a
single ``lgb.Dataset`` whose label is swapped per column (``set_label``), so the
per-label cost is training only, not re-binning a wide float32 matrix ~100-250
times. Columns with fewer than ``GBDT_MIN_POSITIVES`` positives (or negatives)
in the train fold cannot be fit by LightGBM and fall back to a constant score
equal to the train prior.

Feature attribution
-------------------
Per-booster gain importances are summed across labels into one per-feature
vector. When ``ESM_DIM`` is present in config (the driver sets it for the
ESM ⊕ FV arm) the vector is split into an ESM block and an FV block and reported
as gain shares — a direct read on how much predictive work the classical
features are doing once they sit next to the embedding.
"""

from __future__ import annotations

import os
import time
import joblib
import numpy as np
from typeguard import typechecked
from sklearn.model_selection import train_test_split

from protcast.model.stats.utils import calculate_fmax, calculate_smin
from protcast.model.multilabel_classifier import GOEncoder

try:  # keep the module importable when lightgbm is absent (driver's other arms)
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in envs without lightgbm
    lgb = None
    _LGB_AVAILABLE = False


_INSTALL_HINT = (
    "lightgbm is not installed. Locally: `pip install lightgbm` (macOS also needs "
    "`brew install libomp`). On Frontera, inside the TF container: "
    "`pip3 install --user lightgbm` (lands in ~/.local, already on PYTHONPATH)."
)

ARTIFACT_FORMAT = "lightgbm_model_strings_v1"


@typechecked
class GBDTClassifier:
    """One-vs-rest LightGBM multi-label classifier for protein GO prediction.

    Produces the same metrics (Fmax, Smin) and logs to the same MLflow schema
    as KNNClassifier / MultiLabelClassifier for head-to-head comparison.

    Parameters
    ----------
    verbose : bool
        Whether to print progress information.
    protein_embeddings : dict
        {protein_id: np.ndarray} feature vectors (raw ESM, or raw ESM ⊕ FV).
    protein_go_terms : dict
        {protein_id: set[str]} GO term annotations per protein.
    go_ids : list
        GO term IDs to predict (label columns; sorted internally).
    config : dict
        Configuration dictionary from config.json. GBDT_* keys are optional;
        class defaults apply when absent. ESM_DIM (optional) marks where the
        ESM block ends and the FV block begins for importance attribution.
    id : str
        Identifier for this model run (used in filenames).
    use_mlflow : bool
        Whether to log to MLflow.
    go_dag : object or None
        GO DAG for depth-level metric breakdowns. Optional.
    random_state : int
        Seed for the train/val split and LightGBM's bagging / feature sampling.
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
        random_state: int = 42,
    ) -> None:
        self.verbose = verbose
        self.protein_embeddings = protein_embeddings
        self.protein_go_terms = protein_go_terms
        # sorted + de-duplicated so label columns always match GOEncoder's map
        self.go_ids = sorted(set(go_ids))
        self.use_mlflow = use_mlflow
        self.id = id
        self.go_dag = go_dag

        # Set instance attributes from config (same splat as KNN / MLP)
        self.params = config
        for key, value in config.items():
            setattr(self, key.lower(), value)

        # GBDT-specific defaults (overridden by config if present)
        defaults = {
            "gbdt_n_estimators": 300,
            "gbdt_learning_rate": 0.05,
            "gbdt_num_leaves": 31,
            "gbdt_min_child_samples": 20,
            "gbdt_colsample_bytree": 0.3,
            "gbdt_subsample": 0.8,
            "gbdt_reg_lambda": 1.0,
            "gbdt_n_jobs": 0,
            "gbdt_early_stopping_rounds": 0,
            "gbdt_min_positives": 2,
            "gbdt_max_bin": 255,
        }
        for attr, default in defaults.items():
            if not hasattr(self, attr):
                setattr(self, attr, default)

        # Block boundary for importance attribution (None = single block).
        self.esm_dim = getattr(self, "esm_dim", None)
        self.fv_dim = 0

        # Optional extras persisted in the artifact by the driver so downstream
        # analysis (measure_hierarchy_violations.py) can rebuild the exact X
        # without recomputing classical feature vectors.
        self.fv_matrix = None
        self.feature_algorithms = None

        self.random_state = random_state
        self.training_time = 0
        self.logging_time = 0

        self.model_strings = []
        self.constant_cols = {}
        self.best_iterations = []
        self.gain_importance = None
        self.split_importance = None
        self._boosters = None

        self._mlflow = None
        if self.use_mlflow:
            import mlflow as _mlflow_module

            # Reuse the already-initialised module if a parent run is active;
            # re-running init_mlflow() would invalidate the parent handle.
            if _mlflow_module.active_run() is not None:
                self._mlflow = _mlflow_module
            else:
                from protcast.utils.mlflow_utils import init_mlflow
                self._mlflow = init_mlflow(
                    experiment_name=config.get(
                        "EXPERIMENT_NAME", "Default Experiment"
                    ),
                    repo_owner=config.get("DAGSHUB_REPO_OWNER", "aakpan"),
                    repo_name=config.get("DAGSHUB_REPO_NAME", "my-first-repo"),
                    verbose=self.verbose,
                )

    # ------------------------------------------------------------------ run --
    @typechecked
    def run(self) -> None:
        """Main training + evaluation orchestration."""
        self.start_time = time.time()

        if self.use_mlflow and self._mlflow is not None:
            run_name = f"gbdt_{self.id}"
            parent_active = self._mlflow.active_run() is not None
            self._mlflow.start_run(run_name=run_name, nested=parent_active)

        self.prepare_data()
        self.build_model()
        self.train_model()
        if self.use_mlflow:
            self.log_model()

    # --------------------------------------------------------- prepare_data --
    @typechecked
    def prepare_data(self) -> None:
        """Build feature matrix X and multi-hot label matrix y.

        Identical to KNNClassifier / MultiLabelClassifier.prepare_data() —
        same sorted protein set, same column order, same split — so all arms
        see exactly the same data.
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

        if self.esm_dim is not None:
            self.esm_dim = int(self.esm_dim)
            if not 0 < self.esm_dim <= self.X.shape[1]:
                raise ValueError(
                    f"ESM_DIM={self.esm_dim} is outside the feature width "
                    f"{self.X.shape[1]}"
                )
            self.fv_dim = int(self.X.shape[1] - self.esm_dim)
        else:
            self.fv_dim = 0

        if self.verbose:
            num_annotations = int(self.y.sum())
            avg_labels = num_annotations / len(protein_ids)
            print(f"Proteins: {len(protein_ids)}")
            print(f"GO terms: {len(self.go_ids)}")
            print(f"Total annotations: {num_annotations}")
            print(f"Avg GO terms per protein: {avg_labels:.1f}")
            print(f"Embedding dim: {embedding_dim}"
                  + (f"  (ESM {self.esm_dim} + FV {self.fv_dim})"
                     if self.esm_dim is not None else ""))
            print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")

    # ---------------------------------------------------------- build_model --
    @typechecked
    def build_model(self) -> None:
        """Assemble the LightGBM parameter dict (one set for every label)."""
        if not _LGB_AVAILABLE:
            raise ImportError(_INSTALL_HINT)

        self.lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": float(self.gbdt_learning_rate),
            "num_leaves": int(self.gbdt_num_leaves),
            "min_data_in_leaf": int(self.gbdt_min_child_samples),
            "feature_fraction": float(self.gbdt_colsample_bytree),
            "bagging_fraction": float(self.gbdt_subsample),
            "bagging_freq": 1,
            "lambda_l2": float(self.gbdt_reg_lambda),
            "num_threads": int(self.gbdt_n_jobs),
            "max_bin": int(self.gbdt_max_bin),
            "seed": int(self.random_state),
            # Wide float matrices: column-wise histograms, and skip LightGBM's
            # per-fit row/col auto-benchmark (it would run once per label).
            "force_col_wise": True,
            # Required to reuse ONE constructed Dataset across labels; the
            # pre-filter depends on the label and cannot change post-construct.
            "feature_pre_filter": False,
            "verbose": -1,
        }

        if self.verbose:
            es = int(self.gbdt_early_stopping_rounds)
            print(
                f"GBDT config: n_estimators={self.gbdt_n_estimators}, "
                f"lr={self.gbdt_learning_rate}, num_leaves={self.gbdt_num_leaves}, "
                f"min_child={self.gbdt_min_child_samples}, "
                f"colsample={self.gbdt_colsample_bytree}, "
                f"subsample={self.gbdt_subsample}, lambda_l2={self.gbdt_reg_lambda}, "
                f"threads={self.gbdt_n_jobs or 'all'}, "
                f"early_stopping={'off' if es <= 0 else es}"
            )

    # ---------------------------------------------------------- train_model --
    def train_model(self) -> None:
        """Fit one booster per GO term on the train fold; evaluate on val.

        Uses the same train/test split as the other arms (random_state=seed).
        """
        if self.verbose:
            print("Training one-vs-rest LightGBM boosters...")

        validation_split = getattr(self, "validation_split", 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y,
            test_size=validation_split,
            random_state=self.random_state,
        )

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

        n_train, n_labels = y_train.shape
        n_features = X_train.shape[1]
        n_rounds = int(self.gbdt_n_estimators)
        es_rounds = int(self.gbdt_early_stopping_rounds)
        min_pos = int(self.gbdt_min_positives)

        train_start = time.time()

        # Bin the design matrix ONCE; swap labels per column.
        # C-contiguous copies: train_test_split hands back index-selected
        # views that LightGBM would otherwise copy per label (with a warning).
        X_train_c = np.ascontiguousarray(X_train, dtype=np.float32)
        X_val_c = np.ascontiguousarray(X_val, dtype=np.float32)
        # Labels are column slices of a 2-D matrix (non-contiguous); LightGBM
        # copies those with a warning, so hand it contiguous 1-D arrays.
        def _col(y, j):
            return np.ascontiguousarray(y[:, j], dtype=np.float32)

        train_ds = lgb.Dataset(
            X_train_c, label=_col(y_train, 0), params=self.lgb_params,
            free_raw_data=False,
        ).construct()
        val_ds = None
        if es_rounds > 0:
            val_ds = lgb.Dataset(
                X_val_c, label=_col(y_val, 0), reference=train_ds,
                params=self.lgb_params, free_raw_data=False,
            ).construct()

        self.model_strings = [None] * n_labels
        self.constant_cols = {}
        self.best_iterations = [0] * n_labels
        gain = np.zeros(n_features, dtype=np.float64)
        split = np.zeros(n_features, dtype=np.float64)
        y_val_pred = np.zeros((X_val.shape[0], n_labels), dtype=np.float32)
        pos_counts = y_train.sum(axis=0)

        for j in range(n_labels):
            n_pos = int(pos_counts[j])
            n_neg = n_train - n_pos
            if n_pos < min_pos or n_neg < min_pos:
                # LightGBM cannot fit a (near-)single-class target; the honest
                # score is the train prior, constant across proteins.
                prior = float(y_train[:, j].mean())
                self.constant_cols[j] = prior
                y_val_pred[:, j] = prior
                continue

            train_ds.set_label(_col(y_train, j))
            train_kwargs = {}
            if val_ds is not None:
                val_ds.set_label(_col(y_val, j))
                train_kwargs = dict(
                    valid_sets=[val_ds],
                    callbacks=[lgb.early_stopping(
                        es_rounds, first_metric_only=True, verbose=False
                    )],
                )

            booster = lgb.train(
                self.lgb_params, train_ds, num_boost_round=n_rounds, **train_kwargs
            )

            # With early stopping best_iteration > 0 and predict / importance /
            # model_to_string default to it; without, they use every round.
            best_it = int(booster.best_iteration) if booster.best_iteration else 0
            self.best_iterations[j] = best_it if best_it > 0 else booster.current_iteration()

            y_val_pred[:, j] = booster.predict(X_val_c).astype(np.float32)
            gain += booster.feature_importance(importance_type="gain")
            split += booster.feature_importance(importance_type="split")
            self.model_strings[j] = booster.model_to_string()
            booster.free_dataset()

            if self.verbose and ((j + 1) % 10 == 0 or j + 1 == n_labels):
                print(f"  label {j + 1}/{n_labels}  "
                      f"elapsed {time.time() - train_start:.0f}s")

        del train_ds, val_ds, X_train_c, X_val_c
        self._boosters = None  # rebuilt lazily from model_strings on predict()

        self.training_time = time.time() - train_start
        self.gain_importance = gain
        self.split_importance = split
        self.y_val_pred = y_val_pred

        fmax, fmax_threshold = calculate_fmax(y_val, y_val_pred)
        smin, smin_threshold = calculate_smin(y_val, y_val_pred)

        self.best_fmax = fmax
        self.best_threshold = fmax_threshold
        self.best_smin = smin
        self.smin_threshold = smin_threshold

        if self.verbose:
            print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
            print(f"Boosters trained: {n_labels - len(self.constant_cols)}  "
                  f"constant (prior) labels: {len(self.constant_cols)}")
            print(f"GBDT training + eval completed in {self.training_time:.2f}s")
            print(f"Best Fmax: {fmax:.4f} (threshold={fmax_threshold:.2f})")
            print(f"Best Smin: {smin:.4f} (threshold={smin_threshold:.2f})")
            shares = self.compute_block_shares()
            if shares.get("fv_gain_share") is not None:
                print(f"Gain share — ESM: {shares['esm_gain_share']:.3f}  "
                      f"FV: {shares['fv_gain_share']:.3f}  "
                      f"(FV per-dim ratio {shares['fv_gain_per_dim_ratio']:.2f})")

    # -------------------------------------------------------------- predict --
    @staticmethod
    def _predict_with(model_strings, constant_cols, X, boosters=None):
        """Score X with per-label boosters (rebuilt from text) + constants.

        Returns (scores, boosters) so callers can cache the rebuilt boosters.
        """
        if not _LGB_AVAILABLE:
            raise ImportError(_INSTALL_HINT)
        n_labels = len(model_strings)
        if boosters is None:
            boosters = [
                lgb.Booster(model_str=s) if s is not None else None
                for s in model_strings
            ]
        X = np.ascontiguousarray(X, dtype=np.float32)
        scores = np.zeros((X.shape[0], n_labels), dtype=np.float32)
        for j, booster in enumerate(boosters):
            if booster is None:
                scores[:, j] = float(constant_cols.get(j, constant_cols.get(str(j), 0.0)))
            else:
                scores[:, j] = booster.predict(X).astype(np.float32)
        return scores, boosters

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict multi-label scores for a feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Raw feature matrix, shape (n_samples, vector_length).

        Returns
        -------
        np.ndarray
            Score matrix, shape (n_samples, n_classes), values in [0, 1].
        """
        scores, self._boosters = self._predict_with(
            self.model_strings, self.constant_cols, X, self._boosters
        )
        return scores

    # ------------------------------------------------------ block importance --
    def compute_block_shares(self) -> dict:
        """Aggregate per-feature gain into ESM-block vs FV-block shares.

        Returns a JSON-safe dict. When ESM_DIM is unset the whole vector is one
        block and the FV fields are None / empty.
        """
        g = self.gain_importance
        if g is None:
            return {}
        total = float(g.sum())
        esm_dim = int(self.esm_dim) if self.esm_dim is not None else int(g.shape[0])
        fv_dim = int(g.shape[0] - esm_dim)
        esm_gain = float(g[:esm_dim].sum())
        fv_gain = float(g[esm_dim:].sum()) if fv_dim > 0 else 0.0

        out = {
            "esm_dim": esm_dim,
            "fv_dim": fv_dim,
            "esm_gain_share": (esm_gain / total) if total > 0 else None,
            "fv_gain_share": (fv_gain / total) if (total > 0 and fv_dim > 0) else None,
            # Size-normalised: gain per FV dim relative to gain per ESM dim.
            # 1.0 means an FV dimension is, on average, as useful as an ESM one.
            "fv_gain_per_dim_ratio": (
                (fv_gain / fv_dim) / (esm_gain / esm_dim)
                if (fv_dim > 0 and esm_gain > 0) else None
            ),
            "fv_gain_by_feature": (
                [float(v / total) for v in g[esm_dim:]]
                if (fv_dim > 0 and total > 0) else []
            ),
            "fv_split_share": None,
        }
        s = self.split_importance
        if s is not None and fv_dim > 0 and float(s.sum()) > 0:
            out["fv_split_share"] = float(s[esm_dim:].sum() / s.sum())

        # Cheap bonus for mean_max_std pooling of a 1152-d model: gain share of
        # the mean / max / std thirds of the ESM block.
        if esm_dim % 3 == 0 and esm_dim // 3 == 1152 and total > 0:
            thirds = g[:esm_dim].reshape(3, -1).sum(axis=1) / total
            out["esm_pool_block_gain_shares"] = {
                "mean": float(thirds[0]), "max": float(thirds[1]), "std": float(thirds[2]),
            }
        return out

    def save_importances(self, prefix: str | None = None) -> list:
        """Write the per-feature gain / split importance vectors as .npy files.

        Returns the list of written paths. Kept out of the results JSON because
        the vectors are as wide as the input (1152–3468 floats).
        """
        if self.gain_importance is None:
            return []
        prefix = prefix or self.get_name()
        paths = []
        for tag, vec in (("gain", self.gain_importance), ("split", self.split_importance)):
            path = f"{prefix}_{tag}_importance.npy"
            np.save(path, np.asarray(vec, dtype=np.float64))
            paths.append(path)
        if self.verbose:
            print(f"Importances saved to {', '.join(paths)}")
        return paths

    # -------------------------------------------------------------- metrics --
    def compute_depth_metrics(self, y_true, y_pred):
        """Compute Fmax broken down by GO term depth in the DAG.

        Returns
        -------
        dict
            {depth: {"fmax": float, "threshold": float, "n_terms": int,
                      "avg_train_count": float}}
        """
        if self.go_dag is None:
            return {}

        term_depths = {}
        for i, go_id in enumerate(self.go_ids):
            if go_id in self.go_dag.go_terms_map:
                term = self.go_dag.go_terms_map[go_id]
                term_depths[i] = term.depth

        from collections import defaultdict
        depth_groups = defaultdict(list)
        for idx, depth in term_depths.items():
            depth_groups[depth].append(idx)

        results = {}
        train_counts = self.y_train.sum(axis=0) if hasattr(self, "y_train") else self.y.sum(axis=0)

        for depth, term_indices in sorted(depth_groups.items()):
            y_true_subset = y_true[:, term_indices]
            y_pred_subset = y_pred[:, term_indices]

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

    # ---------------------------------------------------------- persistence --
    def hyperparams(self) -> dict:
        """JSON-safe dict of the GBDT hyper-parameters actually used."""
        return {
            "n_estimators": int(self.gbdt_n_estimators),
            "learning_rate": float(self.gbdt_learning_rate),
            "num_leaves": int(self.gbdt_num_leaves),
            "min_child_samples": int(self.gbdt_min_child_samples),
            "colsample_bytree": float(self.gbdt_colsample_bytree),
            "subsample": float(self.gbdt_subsample),
            "reg_lambda": float(self.gbdt_reg_lambda),
            "early_stopping_rounds": int(self.gbdt_early_stopping_rounds),
            "min_positives": int(self.gbdt_min_positives),
            "max_bin": int(self.gbdt_max_bin),
        }

    def save_model(self, path: str | None = None) -> str:
        """Save the trained boosters and metadata to disk (joblib, compressed).

        Boosters are stored as LightGBM text models — portable across LightGBM
        versions and free of C-handle pickling surprises. ``protein_ids`` and
        the optional ``fv_matrix`` let downstream scripts rebuild the exact
        design matrix and split without recomputing classical feature vectors.

        Returns the path to the saved file.
        """
        if path is None:
            path = f"{self.get_name()}.joblib"

        artifact = {
            "format": ARTIFACT_FORMAT,
            "model_strings": self.model_strings,
            "constant_cols": {int(k): float(v) for k, v in self.constant_cols.items()},
            "go_ids": self.go_ids,
            "best_threshold": self.best_threshold,
            "best_fmax": self.best_fmax,
            "vector_length": self.vector_length,
            "esm_dim": self.esm_dim,
            "fv_dim": self.fv_dim,
            "protein_ids": list(self.protein_ids),
            "fv_matrix": (
                np.asarray(self.fv_matrix, dtype=np.float32)
                if self.fv_matrix is not None else None
            ),
            "feature_algorithms": (
                list(self.feature_algorithms)
                if self.feature_algorithms is not None else None
            ),
            "gain_importance": self.gain_importance,
            "split_importance": self.split_importance,
            "best_iterations": list(self.best_iterations),
            "hyperparams": self.hyperparams(),
        }

        joblib.dump(artifact, path, compress=3)
        if self.verbose:
            print(f"Model saved to {path}")
        return path

    @classmethod
    def load_model(cls, path: str) -> dict:
        """Load a trained GBDT artifact from disk (see save_model for keys)."""
        return joblib.load(path)

    @classmethod
    def predict_from_artifact(cls, artifact: dict, X: np.ndarray) -> np.ndarray:
        """Predict using a loaded model artifact.

        Parameters
        ----------
        artifact : dict
            Loaded via GBDTClassifier.load_model().
        X : np.ndarray
            RAW feature matrix, shape (n_samples, vector_length), with the same
            column layout the model was trained on (ESM first, then FV).

        Returns
        -------
        np.ndarray
            Score matrix, shape (n_samples, n_classes).
        """
        scores, _ = cls._predict_with(
            artifact["model_strings"], artifact.get("constant_cols", {}), X
        )
        return scores

    # ------------------------------------------------------------ log_model --
    @typechecked
    def log_model(self) -> None:
        """Log model, metrics, and artifacts to MLflow.

        Same schema as KNNClassifier.log_model() so runs are directly
        comparable in the MLflow UI, plus GBDT-specific hyper-parameters and
        block gain shares.
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
        mlflow.log_param("model_type", "gbdt")
        mlflow.log_param("input_source", "esm_embeddings")
        mlflow.log_param("num_classes", len(self.go_ids))
        mlflow.log_param("feature_vector_length", self.vector_length)
        mlflow.log_param("best_threshold", round(self.best_threshold, 3))
        for k, v in self.hyperparams().items():
            mlflow.log_param(f"gbdt_{k}", v)
        if self.esm_dim is not None:
            mlflow.log_param("gbdt_esm_dim", self.esm_dim)
            mlflow.log_param("gbdt_fv_dim", self.fv_dim)
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

        # GBDT-specific: block gain shares and degenerate-label count
        print("  > Logging GBDT attribution metrics...", end="", flush=True)
        shares = self.compute_block_shares()
        for key in ("esm_gain_share", "fv_gain_share", "fv_gain_per_dim_ratio", "fv_split_share"):
            if shares.get(key) is not None:
                mlflow.log_metric(f"gbdt_{key}", round(float(shares[key]), 6))
        mlflow.log_metric("gbdt_n_constant_labels", len(self.constant_cols))
        if self.best_iterations:
            mlflow.log_metric("gbdt_best_iterations_mean", float(np.mean(self.best_iterations)))
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

        # Save and log the model + importance artifacts
        print("  > Saving model artifact...", end="", flush=True)
        model_path = self.save_model()
        mlflow.log_artifact(model_path, artifact_path="model")
        for p in self.save_importances():
            mlflow.log_artifact(p, artifact_path="importance")
        print(" done.")

        mlflow.set_tag("Training Info", "GBDTClassifier full logging")
        mlflow.set_tag("model_type", "gbdt")

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
        return f"{self.id}_gbdt"

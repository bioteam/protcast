import os
import keras
import pickle
import numpy as np
import time
from pathlib import Path
from typeguard import typechecked
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, History  # type: ignore
from keras.metrics import F1Score  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from protein_feature_vectors import Calculator

os.environ["KERAS_BACKEND"] = "tensorflow"


@typechecked
class MultiClassifier:
    """MultiClassifier
    This class uses feature vectors to train and test a multi-class classifier.
    It uses the Keras Sequential API.
    """

    def __init__(
        self,
        algorithm: str,
        verbose: bool,
        proteins: dict,
        config: dict,
        id: str,
        use_mlflow: bool = False,
        use_tensorboard: bool = False,
        input_source: str = "feature_vectors",
        feature_algorithms: list | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.verbose = verbose
        self.proteins = proteins
        self.use_mlflow = use_mlflow
        self.use_tensorboard = use_tensorboard
        self.id = id
        self.input_source = input_source
        self.feature_algorithms = feature_algorithms or [
            "CTriad",
            "Moran",
            "CTDD",
        ]

        # Validate input_source parameter
        valid_sources = ["feature_vectors", "esm_embeddings", "combined"]
        if input_source not in valid_sources:
            raise ValueError(
                f"input_source must be one of {valid_sources}, got '{input_source}'"
            )

        # Set instance attributes to the values from "config.json"
        self.params = config
        for key, value in config.items():
            setattr(self, key.lower(), value)

        # Initialize data structures
        self.go_ids = list()
        self.pids = list()
        self.features = list()

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
        """run
        Key methods
        """
        self.start_time = time.time()
        self.get_feature_vectors()
        self.prepare_data()
        self.build_model()
        self.train_model()
        if self.input_source == "combined":
            self.save_scalers()
        if self.use_mlflow:
            self.log_model()

    @typechecked
    def get_feature_vectors(self) -> None:
        """get_feature_vectors
        Memory-efficient implementation to create feature vectors or process ESM embeddings.
        Processes each GO ID separately and cleans up memory as it goes.

        Returns feature vectors as a list of lists (per GO id),
        protein ids as a list of lists (per GO id), and GO ids as a list.
        """
        if self.input_source == "combined":
            self.get_combined_features()
        elif self.input_source == "esm_embeddings":
            self.get_esm_embeddings()
        else:
            self.get_traditional_feature_vectors()

    @typechecked
    def get_traditional_feature_vectors(self) -> None:
        """get_traditional_feature_vectors
        Original implementation for traditional feature vectors using Calculator.
        """
        fv = Calculator(verbose=self.verbose)
        go_id_count = len(self.proteins.keys())

        if self.verbose:
            print(f"Processing feature vectors for {go_id_count} GO terms")

        for i, go_id in enumerate(self.proteins.keys()):
            if self.verbose:
                print(
                    f"GO id ({i+1}/{go_id_count}): {go_id} - {len(self.proteins[go_id])} proteins"
                )

            # Process this GO term
            fv.get_feature_vectors(self.algorithm, pdict=self.proteins[go_id])

            # Validate encodings
            if fv.encodings is None:
                raise ValueError(f"No encodings generated for GO id {go_id}.")

            # Extract data from the DataFrame (more memory efficient than DataFrame operations)
            pids = [x[0] for x in fv.encodings.iterrows()]
            # Use a list comprehension with explicit float32 casting for memory efficiency
            vals = [
                x[1].astype(np.float32).tolist()
                for x in fv.encodings.iterrows()
            ]

            # Store data
            self.pids.append(pids)
            self.features.append(vals)
            self.go_ids.append(go_id)

            # Record vector length from the first GO term
            if i == 0:
                self.vector_length = len(self.features[0][0])
                if self.verbose:
                    print(f"Feature vector length: {self.vector_length}")

            # Force cleanup after each GO term
            import gc

            gc.collect()

        # Final cleanup
        del fv
        gc.collect()

    @typechecked
    def get_esm_embeddings(self) -> None:
        """get_esm_embeddings
        Memory-efficient implementation to process ESM embeddings.
        Processes each GO ID separately and cleans up memory as it goes.

        ESM embeddings should be provided as a dictionary mapping protein_id -> numpy array.
        """

        go_id_count = len(self.proteins.keys())

        if self.verbose:
            print(f"Processing ESM embeddings for {go_id_count} GO terms")

        for i, go_id in enumerate(self.proteins.keys()):
            if self.verbose:
                print(
                    f"GO id ({i+1}/{go_id_count}): {go_id} - {len(self.proteins[go_id])} proteins"
                )

            # Get ESM embeddings for proteins in this GO term
            pids = []
            vals = []

            for protein_id in self.proteins[go_id]:
                pids.append(protein_id)
                # Convert numpy array to list with float32 for consistency
                embedding = self.proteins[go_id][protein_id]
                if hasattr(embedding, "astype"):
                    vals.append(embedding.astype(np.float32).tolist())
                else:
                    vals.append([float(x) for x in embedding])

            if not pids:
                raise ValueError(
                    f"No ESM embeddings found for any proteins in GO term {go_id}"
                )

            # Store data (same format as traditional feature vectors)
            self.pids.append(pids)
            self.features.append(vals)
            self.go_ids.append(go_id)

            # Record vector length from the first GO term
            if i == 0:
                self.vector_length = len(self.features[0][0])
                if self.verbose:
                    print(f"ESM embedding length: {self.vector_length}")

            # Verify all embeddings have the same length
            elif len(self.features[i][0]) != self.vector_length:
                raise ValueError(
                    f"Inconsistent embedding dimensions: expected {self.vector_length}, "
                    f"got {len(self.features[i][0])} for GO term {go_id}"
                )

            # Force cleanup after each GO term
            import gc

            gc.collect()

        if self.verbose:
            total_proteins = sum(len(pids) for pids in self.pids)
            print(
                f"Successfully processed {total_proteins} proteins with ESM embeddings"
            )

    @typechecked
    def get_combined_features(self) -> None:
        """get_combined_features
        Combine ESM-C embeddings with traditional feature vectors.

        Expects self.proteins to be structured as:
            {go_id: {protein_id: {"embedding": np.array, "sequence": str}}}

        ESM embeddings and feature vectors are normalized separately using
        StandardScaler before concatenation, preventing the higher-dimensional
        ESM embeddings from dominating the feature space.
        """
        import gc

        go_id_count = len(self.proteins.keys())
        fv = Calculator(verbose=self.verbose)

        if self.verbose:
            print(f"Processing combined features for {go_id_count} GO terms")
            print(f"Feature algorithms: {self.feature_algorithms}")

        # First pass: collect all embeddings and feature vectors per GO term
        all_embeddings = []  # list of lists (per GO term)
        all_fv_vectors = []  # list of lists (per GO term)

        for i, go_id in enumerate(self.proteins.keys()):
            protein_data = self.proteins[go_id]
            if self.verbose:
                print(
                    f"GO id ({i+1}/{go_id_count}): {go_id} - {len(protein_data)} proteins"
                )

            pids = []
            embeddings = []
            sequences = {}

            for protein_id, data in protein_data.items():
                pids.append(protein_id)
                embedding = data["embedding"]
                if hasattr(embedding, "astype"):
                    embeddings.append(embedding.astype(np.float32))
                else:
                    embeddings.append(np.array(embedding, dtype=np.float32))
                sequences[protein_id] = data["sequence"]

            if not pids:
                raise ValueError(
                    f"No proteins found for GO term {go_id}"
                )

            # Compute traditional feature vectors for all selected algorithms
            fv_parts = []
            for algo in self.feature_algorithms:
                fv.get_feature_vectors(algo, pdict=sequences)
                if fv.encodings is None:
                    raise ValueError(
                        f"No {algo} encodings generated for GO id {go_id}."
                    )
                # Extract values in the same protein order
                algo_vectors = []
                for pid in pids:
                    if pid in fv.encodings.index:
                        algo_vectors.append(
                            fv.encodings.loc[pid].values.astype(np.float32)
                        )
                    else:
                        raise ValueError(
                            f"Protein {pid} not found in {algo} encodings for GO id {go_id}"
                        )
                fv_parts.append(np.array(algo_vectors, dtype=np.float32))

            # Concatenate all feature algorithm vectors: [n_proteins, total_fv_dim]
            fv_combined = np.hstack(fv_parts)

            self.pids.append(pids)
            self.go_ids.append(go_id)
            all_embeddings.append(np.array(embeddings, dtype=np.float32))
            all_fv_vectors.append(fv_combined)

            if i == 0:
                self.embedding_dim = embeddings[0].shape[0]
                self.fv_dim = fv_combined.shape[1]
                if self.verbose:
                    print(f"ESM embedding dimension: {self.embedding_dim}")
                    print(f"Feature vector dimension: {self.fv_dim}")

            gc.collect()

        # Stack all data across GO terms for scaler fitting
        all_emb_array = np.vstack(all_embeddings)
        all_fv_array = np.vstack(all_fv_vectors)

        # Fit separate scalers for embeddings and feature vectors
        self.emb_scaler = StandardScaler()
        self.fv_scaler = StandardScaler()
        all_emb_scaled = self.emb_scaler.fit_transform(all_emb_array)
        all_fv_scaled = self.fv_scaler.fit_transform(all_fv_array)

        # Concatenate scaled features
        all_combined = np.hstack(
            [all_emb_scaled, all_fv_scaled]
        ).astype(np.float32)

        if self.verbose:
            print(
                f"Combined feature dimension: {all_combined.shape[1]} "
                f"(ESM: {self.embedding_dim} + FV: {self.fv_dim})"
            )

        # Redistribute into per-GO-term lists to match expected format
        offset = 0
        for i in range(len(self.go_ids)):
            n = len(self.pids[i])
            self.features.append(all_combined[offset : offset + n].tolist())
            offset += n

        self.vector_length = all_combined.shape[1]

        # Clean up
        del all_emb_array, all_fv_array, all_emb_scaled, all_fv_scaled, all_combined
        del fv, all_embeddings, all_fv_vectors
        gc.collect()

        if self.verbose:
            total_proteins = sum(len(pids) for pids in self.pids)
            print(
                f"Successfully processed {total_proteins} proteins with combined features"
            )

    def save_scalers(self) -> None:
        """Save the embedding and feature vector scalers for inference."""
        scalers = {
            "emb_scaler": self.emb_scaler,
            "fv_scaler": self.fv_scaler,
            "feature_algorithms": self.feature_algorithms,
            "embedding_dim": self.embedding_dim,
            "fv_dim": self.fv_dim,
        }
        filename = f"{self.get_name()}_scalers.pkl"
        with open(filename, "wb") as f:
            pickle.dump(scalers, f)
        if self.verbose:
            print(f"Scalers saved to {filename}")

    @staticmethod
    def load_scalers(filename: str) -> dict:
        """Load scalers from a pickle file for inference."""
        with open(filename, "rb") as f:
            return pickle.load(f)

    @typechecked
    def prepare_data(self) -> None:
        """prepare_data

        Memory-efficient implementation that processes data in batches
        to avoid large memory allocations.
        """
        # Create GO encoder first (needed for all batches)
        go_encoder = GOEncoder(self.id)
        go_encoder.fit(self.go_ids)
        go_encoder.save()

        # Calculate total samples for informational purposes
        num_samples = sum(len(go_set) for go_set in self.features)
        if self.verbose:
            print(f"Total samples across all GO terms: {num_samples}")
            print(f"Vector length per sample: {self.vector_length}")
            print(f"Number of GO classes: {len(self.go_ids)}")

        # Process each GO term separately
        X_batches = []
        y_batches = []

        for i, (go_id, go_set) in enumerate(zip(self.go_ids, self.features)):
            if self.verbose:
                print(
                    f"Processing GO term {i+1}/{len(self.go_ids)}: {go_id} with {len(go_set)} samples"
                )

            # Create batch for current GO term
            X_batch = np.array(
                go_set, dtype=np.float32
            )  # Use float32 instead of float64 to save memory
            y_integer = go_encoder.encode(go_id)
            y_batch = np.full(len(go_set), y_integer, dtype=np.int32)

            X_batches.append(X_batch)
            y_batches.append(y_batch)

        # Concatenate all batches (more memory efficient than pre-allocating a huge array)
        if self.verbose:
            print("Combining batches...")

        self.X = np.vstack(X_batches)
        y_combined = np.concatenate(y_batches)

        # Convert integers into a binary matrix (one-hot encoding)
        self.y = to_categorical(y_combined, num_classes=len(self.go_ids))

        if self.verbose:
            print(f"Final shape of self.X: {self.X.shape}")
            print(f"Final shape of self.y: {self.y.shape}")
            print(
                f"Memory usage estimate: {self.X.nbytes / 1e9:.2f} GB + {self.y.nbytes / 1e9:.2f} GB"
            )

    @typechecked
    def build_model(self) -> None:
        """build_model

        The final layer uses num_classes for the number of units, ensuring it matches the number
        of GO ids. The model is compiled with 'categorical_crossentropy' loss, which is suitable for
        multi-class problems with mutually exclusive classes.
        """
        # X.shape[1] = self.vector_length, X.shape[0] = total number of samples across GO ids.
        input_shape = (self.X.shape[1],)

        """
        Both relu and softmax activations can be used in a multi-classification model. 
        This is a common and effective approach in neural network architectures. 

        ReLU (Rectified Linear Unit) Activation:

        Used in hidden layers of the network.
        Formula: f(x) = max(0, x)
        Advantages:
        - Helps with the vanishing gradient problem.
        - Allows for sparse activation (some neurons can be completely off).
        - Computationally efficient.
        - Learns complex non-linear relationships between protein features and functions.

        Softmax Activation:

        Used in the output layer for multi-class classification.
        Converts the raw output scores into probabilities that sum to 1.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j in all classes
        Advantages:
        - Provides a probability distribution over all classes.
        - Suitable for mutually exclusive classes.

        Input (protein features)
                    ↓
        Dense(128) + ReLU  ← Non-linear transformation
                    ↓
        Dropout(0.5)       ← Regularization
                    ↓
        Dense(64) + ReLU   ← More non-linear transformation
                    ↓
        Dropout(0.3)       ← More regularization
                    ↓
        Dense(GO_classes) + Softmax  ← Probability distribution over GO functions
                    ↓
        Output probabilities (sum = 1.0)
        """
        # Use wider layers for combined mode due to larger input dimensionality
        if self.input_source == "combined":
            dense1_units = 256
            dense2_units = 128
        else:
            dense1_units = 128
            dense2_units = 64

        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Dense(dense1_units, activation="relu"),
                layers.Dropout(self.dropout),  # type: ignore
                layers.Dense(dense2_units, activation="relu"),
                layers.Dropout(0.3),  # Could make this configurable too
                layers.Dense(len(self.go_ids), activation="softmax"),
            ]
        )

        # Build metrics list - include both config metrics and F1Score
        metrics_list = list(self.metrics) if isinstance(self.metrics, list) else [self.metrics]  # type: ignore
        # Add F1Score for multi-class classification (macro averaging)
        metrics_list.append(F1Score(average="macro", name="f1_score"))

        model.compile(
            optimizer=self.optimizer,  # type: ignore
            loss=self.loss,  # type: ignore
            metrics=metrics_list,
        )

        self.model = model

    def train_model(self) -> History:
        # Memory-efficient training implementation
        if self.verbose:
            print("Starting memory-efficient training process...")
            print(f"Data shapes - X: {self.X.shape}, y: {self.y.shape}")

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=self.validation_split, stratify=self.y  # type: ignore
        )

        # Store training data for inference signature in MLflow
        self.X_train = X_train

        if self.verbose:
            print(
                f"Training set: {X_train.shape}, Validation set: {X_val.shape}"
            )
            print(
                f"Memory used - Training: {X_train.nbytes / 1e9:.2f} GB, Validation: {X_val.nbytes / 1e9:.2f} GB"
            )

        # Early stopping callback based on F1 score (higher is better)
        early_stopping_f1 = EarlyStopping(
            monitor="val_f1_score",
            min_delta=0.001,  # type: ignore
            patience=self.patience,  # type: ignore
            restore_best_weights=True,
            mode="max",  # F1 score should be maximized
        )

        # Save the best model during training
        loss_checkpoint = ModelCheckpoint(
            filepath=f"{self.get_name()}.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        )

        callbacks = [early_stopping_f1, loss_checkpoint]

        # Add TensorBoard logging if requested
        if self.use_tensorboard:
            log_dir = "logs/fit/" + time.strftime(
                "%m-%d-%Y-%H-%M-%S", time.localtime()
            )
            tensorboard_callback = TensorBoard(
                log_dir=log_dir, histogram_freq=1
            )
            callbacks.append(tensorboard_callback)

        # Start timing for training
        train_start_time = time.time()

        # Determine optimal batch size based on data size
        # Use smaller batches for larger datasets to save memory
        adaptive_batch_size = self.batch_size  # type: ignore
        if X_train.shape[0] > 10000:  # For very large datasets
            adaptive_batch_size = min(
                adaptive_batch_size, 64
            )  # Use smaller batches for large datasets
            if self.verbose:
                print(
                    f"Large dataset detected, using reduced batch size: {adaptive_batch_size}"
                )

        # Fit the model with memory-efficient batch size
        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,  # type: ignore
            batch_size=adaptive_batch_size,  # Dynamic batch size based on data size
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose="auto",
            # New parameters to improve memory efficiency:
            shuffle=True,  # Shuffle data for better training
            use_multiprocessing=False,  # Avoid multiprocessing to reduce memory overhead
            workers=1,  # Single worker to avoid memory duplication
        )

        # End timing and store the duration
        self.training_time = time.time() - train_start_time
        self.history = history

        if self.verbose:
            print(f"Training completed in {self.training_time:.2f} seconds")

        # Help clean up memory after training
        import gc

        gc.collect()

        return history  # type: ignore

    @typechecked
    def save_model(self) -> None:
        """save_model
        Save the trained Keras model to disk.

        This method saves the trained neural network model to a file using Keras'
        native save format (.keras). The model file includes the architecture,
        weights, optimizer state, and compilation configuration, allowing for
        complete model restoration later.

        The saved model filename is automatically generated using a timestamp
        and the algorithm name to ensure uniqueness and traceability.
        Parameters
        ----------
        None
        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If the model has not been built or trained yet (self.model is None).
        OSError
            If there are permission issues or insufficient disk space when
            saving the file.

        Notes
        -----
        The model is saved in Keras' native .keras format, which is recommended
        for new applications. This format preserves:
        - Model architecture
        - Model weights
        - Optimizer state
        - Compilation configuration

        The filename format is: {timestamp}_{algorithm}.keras
        where timestamp follows the pattern MM-DD-YYYY-HH-MM-SS.

        Examples
        --------
        >>> classifier = MultiClassifier(algorithm="aac", verbose=True, proteins=data)
        >>> classifier.run()  # Build and train the model
        >>> classifier.save_model()  # Saves to file like "01-15-2024-14-30-45_aac.keras"

        See Also
        --------
        load_model : Class method to load a saved model from disk
        get_name : Method that generates the filename for the saved model
        """
        self.model.save(f"{self.get_name()}.keras")

    @typechecked
    def log_model(self) -> None:
        # Start timing for logging
        log_start_time = time.time()

        print("\n--- Starting MLflow Logging ---")

        mlflow = self._mlflow
        if mlflow is None:
            if self.verbose:
                print("mlflow not available; skipping logging")
            return

        try:
            from mlflow.tracking import MlflowClient  # type: ignore
            from mlflow.entities import Metric  # type: ignore
            from mlflow.models.signature import infer_signature  # type: ignore
            from protcast.utils.mlflow_utils import save_run_metadata  # type: ignore
        except Exception as e:
            if self.verbose:
                print("mlflow sub-imports failed; skipping logging:", e)
            return

        # --- LOG PARAMETERS ---
        mlflow.log_params(self.params)

        # --- LOG DATASET METADATA ---
        print("  > Logging dataset metadata...", end="", flush=True)
        total_samples = self.X.shape[0]
        train_samples = self.X_train.shape[0]
        val_samples = total_samples - train_samples
        mlflow.log_param("input_source", self.input_source)
        mlflow.log_param("algorithm", self.algorithm)
        mlflow.log_param("num_classes", len(self.go_ids))
        mlflow.log_param("feature_vector_length", self.vector_length)
        mlflow.log_metric("total_samples", total_samples)
        mlflow.log_metric("train_samples", train_samples)
        mlflow.log_metric("val_samples", val_samples)
        print(" done.")

        # --- LOG PER-CLASS SAMPLE COUNTS ---
        print("  > Logging per-class sample counts...", end="", flush=True)
        for i, (go_id, go_set) in enumerate(
            zip(self.go_ids, self.features)
        ):
            mlflow.log_metric(f"samples_{go_id}", len(go_set))
        print(" done.")

        # --- LOG BEST EPOCH AND FINAL VAL METRICS ---
        print("  > Logging best epoch metrics...", end="", flush=True)
        history = self.history.history
        if "val_f1_score" in history:
            best_val_f1 = max(history["val_f1_score"])
            best_epoch = history["val_f1_score"].index(best_val_f1) + 1
            mlflow.log_metric("best_val_f1_score", round(best_val_f1, 4))
            mlflow.log_metric("best_epoch", best_epoch)
        if "val_loss" in history:
            final_val_loss = min(history["val_loss"])
            mlflow.log_metric("best_val_loss", round(final_val_loss, 4))
        if "val_accuracy" in history:
            best_val_acc = max(history["val_accuracy"])
            mlflow.log_metric("best_val_accuracy", round(best_val_acc, 4))
        mlflow.log_metric("epochs_run", len(history.get("loss", [])))
        print(" done.")

        # --- BATCH LOGGING WITH FEEDBACK ---
        print("  > Logging epoch metrics in a single batch...", end="", flush=True)
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        metrics_to_log = []
        num_epochs = len(next(iter(history.values())))

        for epoch in range(num_epochs):
            for metric_name, values in history.items():
                metrics_to_log.append(
                    Metric(
                        key=metric_name,
                        value=values[epoch],
                        timestamp=int(time.time() * 1000),
                        step=epoch,
                    )
                )

        client.log_batch(run_id=run_id, metrics=metrics_to_log)
        print(" done.")

        # --- LOG GO TERM LIST AS ARTIFACT ---
        print("  > Logging GO term list artifact...", end="", flush=True)
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="go_terms_"
        ) as f:
            for go_id in self.go_ids:
                f.write(f"{go_id}\n")
            go_terms_path = f.name
        mlflow.log_artifact(go_terms_path, artifact_path="metadata")
        os.remove(go_terms_path)
        print(" done.")

        # --- SIGNATURE INFERENCE WITH FEEDBACK ---
        print("  > Inferring model signature...", end="", flush=True)
        signature = infer_signature(
            self.X_train, self.model.predict(self.X_train, verbose=0)
        )
        print(" done.")

        # --- MODEL UPLOAD WITH FEEDBACK ---
        registered_name = self.params.get(
            "REGISTERED_MODEL_NAME", "multiclassifier.v0"
        )
        print("  > Uploading model artifact to MLflow...", end="", flush=True)
        mlflow.keras.log_model(  # type: ignore
            self.model,
            artifact_path="model",
            signature=signature,
            registered_model_name=registered_name,
        )
        print(" done.")

        mlflow.set_tag("Training Info", "MultiClassifier full logging")
        mlflow.set_tag("input_source", self.input_source)

        # End timing and store the duration
        self.logging_time = time.time() - log_start_time

        # Log the final summary timing metrics
        mlflow.log_metric(
            "training_time_seconds", round(self.training_time, 2)
        )
        mlflow.log_metric(
            "total_logging_time_seconds", round(self.logging_time, 2)
        )

        # --- SAVE RUN METADATA FOR INFERENCE LINKING ---
        save_run_metadata(
            model_name=self.get_name(),
            run_id=run_id,
            experiment_name=self.params.get(
                "EXPERIMENT_NAME", "Default Experiment"
            ),
        )

        print("--- MLflow Logging Complete ---")

    @classmethod
    def load_model(cls, model_path: Path) -> keras.models.Model:
        """load_model

        Load a Keras model from a file.

        This class method loads a pre-trained Keras model from the specified path.
        It's designed to work with models saved in Keras' native .keras format,
        which includes the architecture, weights, optimizer state, and compilation
        configuration. The loaded model can be used for inference or further training.

        Parameters
        ----------
        model_path : Path
            A file system path to the saved model file (.keras).

        Returns
        -------
        keras.models.Sequential
            A Keras Sequential model instance loaded from the file.
            This is ready to use for predictions, evaluation, or further training.

        Examples
        --------
        >>> from protcast.model.multi_classifier import MultiClassifier

        # Load an existing model from disk
        >>> model_path = Path("/path/to/saved_model.keras")
        >>> loaded_model = MultiClassifier.load_model(model_path)

        Notes
        -----
        The expected format for the saved model is .keras, which preserves:
        - Model architecture
        - Model weights
        - Optimizer state
        - Compilation configuration

        See Also
        --------
        save_model : Class method to save a trained Keras model
        get_name : Method that generates the filename for the saved model
        """
        model = keras.models.load_model(model_path)
        return model  # type: ignore[return]

    @typechecked
    def get_name(self) -> str:
        """get_name

        Generate a unique model name using the current timestamp and algorithm name.

        Returns
        -------
        str
            The generated model name in the format MM-DD-YYYY-HH-MM-SS_algorithm
        """
        return f"{self.id}_{self.algorithm}"


class GOEncoder:
    """GOEncoder

    # Example usage:
    go_encoder = GOEncoder('test')
    go_ids = ['GO:1224', 'GO:5678', 'GO:9101', 'GO:1224', 'GO:5678']
    go_encoder.fit(go_ids)

    # Encode GO IDs
    num = go_encoder.encode('GO:1224')

    # Save the encoder
    go_encoder.save()

    # Load the encoder
    loaded_encoder = GOEncoder.load('go_encoder.pickle')

    # Decode using the loaded encoder
    go_id = loaded_encoder.decode(num)

    # Decode GO ids based on probabilities
    probabilities = np.array([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]], [0.2, 0.2, 0.4]])
    top_go_ids = loaded_encoder.decode_probabilities(probabilities, top_k=2)
    print(f"Top 2 GO IDs with probabilities: {top_go_ids}")
    """

    def __init__(self, id):
        """Initialize a GOEncoder instance.

        Attributes
        ----------
        go_to_int : dict or None
            Mapping from GO ID (str) to integer label.
        int_to_go : dict or None
            Mapping from integer label to GO ID (str).
        num_classes : int
            Number of unique GO IDs.
        id : str
            Identifier for the GOEncoder instance, used in saving the encoder.
        """
        self.id = id
        self.go_to_int = None
        self.int_to_go = None
        self.num_classes = 0

    def fit(self, go_ids):
        """fit
        Map the list of GO IDs to a list of integers, and integers to GO ids.
        """
        unique_go_ids = sorted(set(go_ids))
        self.go_to_int = {go: i for i, go in enumerate(unique_go_ids)}
        self.int_to_go = {i: go for go, i in self.go_to_int.items()}
        self.num_classes = len(unique_go_ids)

    def encode(self, go_id):
        """Encode a GO ID to a category number"""
        if not self.go_to_int:
            raise ValueError("GOEncoder has not been fit to any GO IDs.")
        return self.go_to_int[go_id]

    def decode(self, categorical):
        """Decode categories back to GO IDs."""
        if not self.int_to_go:
            raise ValueError("GOEncoder has not been fit to any GO IDs.")
        # Handle single integer
        if isinstance(categorical, (int, np.integer)):
            key = int(categorical)
            return self.int_to_go.get(key, None)
        else:
            integer_encoded = np.argmax(categorical, axis=1)
            return [self.int_to_go[i] for i in integer_encoded]

    def decode_probabilities(self, probabilities, top_k=1):
        """Decode probability distributions to top k GO IDs with their probabilities."""
        if not self.int_to_go:
            raise ValueError("GOEncoder has not been fit to any GO IDs.")
        # 2D array slicing: Select all rows with ':' then
        # select the last top_k elements of each row
        top_indices = np.argsort(probabilities, axis=1)[:, -top_k:]
        result = []
        for i, indices in enumerate(top_indices):
            go_probs = [
                (self.int_to_go[idx], probabilities[i, idx]) for idx in indices
            ]
            result.append(sorted(go_probs, key=lambda x: x[1], reverse=True))
        return result

    def save(self):
        """save

        Serialize the GOEncoder instance to a file using pickle.

        The filename is generated using the current timestamp.

        Returns
        -------
        None
        """
        filename = f"{self.id}_GOEncoder.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """load

        Deserialize a GOEncoder instance from a pickle file.

        Parameters
        ----------
        filename : str
            Path to the pickle file containing the saved encoder.

        Returns
        -------
        GOEncoder
            The loaded GOEncoder instance.
        """
        with open(filename, "rb") as f:
            encoder = pickle.load(f)
        return encoder

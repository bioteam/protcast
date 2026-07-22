import os
import keras
import pickle
import numpy as np
import time
from pathlib import Path
from typeguard import typechecked
from keras import layers, models
from keras.callbacks import ModelCheckpoint, TensorBoard, History
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from protcast.model.stats.utils import calculate_fmax

os.environ["KERAS_BACKEND"] = "tensorflow"

# Optional order embedding imports — only needed when USE_ORDER_EMBEDDINGS is True
try:
    from protcast.model.order_embeddings import (
        OrderEmbeddingLayer,
        build_order_embedding_model,
        build_order_embedding_model_dual,
        order_violation_loss,
        order_violation_loss_soft,
    )
    from protcast.preprocessing.go_dag_edges import extract_dag_edges

    _ORDER_AVAILABLE = True
except ImportError:
    _ORDER_AVAILABLE = False


class FmaxEarlyStopping(keras.callbacks.Callback):
    """Early stopping callback that monitors CAFA-style Fmax.

    At the end of each epoch, computes Fmax on the validation set by
    sweeping thresholds. Stops training when Fmax hasn't improved for
    `patience` epochs and restores the weights from the best epoch.

    This replaces the standard EarlyStopping(monitor="val_loss") so
    the model trains directly toward the evaluation metric used in CAFA.

    Parameters
    ----------
    X_val : np.ndarray
        Validation feature matrix.
    y_val : np.ndarray
        Validation multi-hot label matrix.
    patience : int
        Number of epochs with no Fmax improvement before stopping.
    min_delta : float
        Minimum improvement to count as progress.
    verbose : bool
        Whether to print Fmax each epoch.
    """

    def __init__(self, X_val, y_val, patience=10, min_delta=0.001, verbose=False):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience
        self.min_delta = min_delta
        self._verbose = verbose

        self.best_fmax = 0.0
        self.best_threshold = 0.5
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        y_pred = self.model.predict(self.X_val, verbose=0)
        fmax, threshold = calculate_fmax(self.y_val, y_pred)

        # Store in logs so other callbacks/history can see it
        logs["val_fmax"] = fmax
        logs["val_fmax_threshold"] = threshold

        if self._verbose:
            print(f"  val_fmax: {fmax:.4f} (threshold={threshold:.2f})", end="")

        if fmax > self.best_fmax + self.min_delta:
            self.best_fmax = fmax
            self.best_threshold = threshold
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.wait = 0
            if self._verbose:
                print(" *")
        else:
            self.wait += 1
            if self._verbose:
                print(f" (no improvement, wait={self.wait}/{self.patience})")
            if self.wait >= self.patience:
                if self._verbose:
                    print(
                        f"Fmax early stopping: best={self.best_fmax:.4f} "
                        f"at epoch {self.best_epoch + 1}"
                    )
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            if self._verbose:
                print(
                    f"Restoring weights from epoch {self.best_epoch + 1} "
                    f"(Fmax={self.best_fmax:.4f}, threshold={self.best_threshold:.2f})"
                )
            self.model.set_weights(self.best_weights)


@typechecked
class MultiLabelClassifier:
    """MultiLabelClassifier

    Multi-label classifier for protein function prediction. Uses sigmoid
    activations so each protein can be assigned multiple GO terms
    simultaneously.

    - Input: {protein_id: embedding_vector} (one entry per protein)
    - Labels: multi-hot vectors (multiple GO terms per protein)
    - Output activation: sigmoid (independent per-class probabilities)
    - Loss: binary_crossentropy
    - Evaluation: Fmax (threshold-optimized F1), protein-centric
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
        use_tensorboard: bool = False,
        go_dag: object = None,
        random_state: int = 42,
        scale_features: bool = False,
    ) -> None:
        """
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
        use_tensorboard : bool
            Whether to log to TensorBoard.
        go_dag : AnnotatedGODag or None
            GO DAG for the order-embedding violation loss. Required when
            USE_ORDER_EMBEDDINGS is True. Ignored for flat model.
        random_state : int
            Random seed for train/test split. Default 42.
        scale_features : bool
            If True, standardize features with StandardScaler fit on the
            training fold only (then applied to validation) inside
            train_model. Default False preserves legacy behavior; callers
            that compare embeddings (e.g. the feature scan) should set True
            so every model is scaled consistently and without leakage.
        """
        self.verbose = verbose
        self.protein_embeddings = protein_embeddings
        self.protein_go_terms = protein_go_terms
        self.go_ids = sorted(go_ids)
        self.use_mlflow = use_mlflow
        self.use_tensorboard = use_tensorboard
        self.id = id
        self.go_dag = go_dag
        self.random_state = random_state
        self.scale_features = scale_features

        # Set instance attributes from config
        self.params = config
        for key, value in config.items():
            setattr(self, key.lower(), value)

        self.training_time = 0
        self.logging_time = 0

        self._mlflow = None
        if self.use_mlflow:
            import mlflow as _mlflow_module

            # If a parent run is already active (a comparison wrapper script
            # has set things up), re-running init_mlflow() would reset client
            # state and invalidate the parent run handle.  Just reuse the
            # already-initialized module instead.
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

    @typechecked
    def run(self) -> None:
        """Main training orchestration."""
        self.start_time = time.time()

        # Start MLflow run before training so all logging goes to one run.
        # If a parent run is already active (e.g. a comparison wrapper),
        # this run is created as nested so the UI groups them together.
        if self.use_mlflow and self._mlflow is not None:
            variant = "order" if getattr(self, "use_order_embeddings", False) else "flat"
            run_name = f"multilabel_{variant}_{self.id}"
            parent_active = self._mlflow.active_run() is not None
            self._mlflow.start_run(run_name=run_name, nested=parent_active)

        self.prepare_data()
        self.build_model()
        self.train_model()
        if self.use_mlflow:
            self.log_model()

    @typechecked
    def prepare_data(self) -> None:
        """Build feature matrix X and multi-hot label matrix y.

        Creates one row per protein. Each protein's label vector has a 1
        for every GO term it is annotated with.
        """
        # Build GO encoder
        go_encoder = GOEncoder(self.id)
        go_encoder.fit(self.go_ids)
        go_encoder.save()
        self.go_encoder = go_encoder

        # Only include proteins that have both embeddings and annotations
        protein_ids = sorted(
            set(self.protein_embeddings.keys()) & set(self.protein_go_terms.keys())
        )

        if not protein_ids:
            raise ValueError(
                "No proteins found with both embeddings and GO annotations."
            )

        # Build X matrix
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

            # Multi-hot encode GO terms for this protein
            label = np.zeros(len(self.go_ids), dtype=np.float32)
            for go_id in self.protein_go_terms[pid]:
                if go_id in go_encoder.go_to_int:
                    label[go_encoder.go_to_int[go_id]] = 1.0
            y_list.append(label)

        self.X = np.vstack(X_list)
        self.y = np.array(y_list)
        self.protein_ids = protein_ids

        # Compute per-class positive weights for imbalance handling.
        # For each GO term, weight = (num_proteins / num_positive) so that
        # rare terms get higher weight. Capped to avoid extreme values.
        pos_counts = self.y.sum(axis=0)  # shape: (num_classes,)
        n_samples = self.y.shape[0]
        # Avoid division by zero for GO terms with no annotations
        pos_counts = np.maximum(pos_counts, 1.0)
        raw_weights = n_samples / pos_counts
        # Cap at 10x to prevent rare terms from dominating
        self.class_weights = np.minimum(raw_weights, 10.0).astype(np.float32)

        if self.verbose:
            num_annotations = int(self.y.sum())
            avg_labels = num_annotations / len(protein_ids)
            print(f"Proteins: {len(protein_ids)}")
            print(f"GO terms: {len(self.go_ids)}")
            print(f"Total annotations: {num_annotations}")
            print(f"Avg GO terms per protein: {avg_labels:.1f}")
            print(f"Embedding dim: {embedding_dim}")
            print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")
            min_w, max_w = self.class_weights.min(), self.class_weights.max()
            print(f"Class weight range: {min_w:.2f} - {max_w:.2f}")

    @typechecked
    def build_model(self) -> None:
        """Build the multi-label neural network.

        Two modes controlled by the USE_ORDER_EMBEDDINGS config key:

        **Flat mode** (default):
            Input → [Dense+Dropout]* → Dense(sigmoid) → scores
            Standard multi-label with weighted binary crossentropy.

        **Order mode** (USE_ORDER_EMBEDDINGS=True):
            Input → [Dense+Dropout]* → Dense(order_dim) → OrderEmbeddingLayer → scores
            Adds an order-violation loss over GO DAG edges so that child terms
            dominate (are more specific than) their parents in the product
            order. Requires a go_dag to be provided at init time.

        **Dual-encoder order mode** (USE_ORDER_EMBEDDINGS=True, USE_DUAL_ENCODER=True):
            ESM-C → [Dense+Dropout]* ──┐
                                        ├─ Concat → Dense(order_dim) → OrderEmbeddingLayer
            FV → Dense(fv_hidden) ─────┘
            Gives the small PseKRAAC block its own MLP branch so its 12 dims
            aren't drowned out by 1152 ESM-C dims in the first weight matrix.
            Requires protein_embeddings values to be dicts with "esm" and "fv" keys,
            or self.X to be set externally as a (N, esm_dim+fv_dim) matrix with
            ESM_DIM config key indicating the split point.

        Architecture is configurable via HIDDEN_LAYERS (list of ints).
        """
        import tensorflow as tf

        input_dim = self.X.shape[1]
        num_classes = len(self.go_ids)
        hidden_layers = getattr(self, "hidden_layers", [128, 64])
        dropout_rate = self.dropout  # type: ignore
        use_order = getattr(self, "use_order_embeddings", False)
        use_dual = use_order and getattr(self, "use_dual_encoder", False)

        class_weights_tensor = tf.constant(
            self.class_weights, dtype=tf.float32
        )

        if use_dual:
            esm_dim = getattr(self, "esm_dim", None)
            if esm_dim is None:
                raise ValueError(
                    "USE_DUAL_ENCODER requires ESM_DIM in config to split the "
                    "concatenated input into ESM-C and feature-vector blocks."
                )
            fv_dim = input_dim - esm_dim
            self._build_order_model_dual(
                esm_dim, fv_dim, num_classes, hidden_layers, dropout_rate,
                class_weights_tensor, tf,
            )
        elif use_order:
            self._build_order_model(
                input_dim, num_classes, hidden_layers, dropout_rate,
                class_weights_tensor, tf,
            )
        else:
            self._build_flat_model(
                input_dim, num_classes, hidden_layers, dropout_rate,
                class_weights_tensor, tf,
            )

        if self.verbose:
            if use_dual:
                mode = "order-dual-encoder"
            elif use_order:
                mode = "order"
            else:
                mode = "flat"
            print(f"Model mode: {mode}")
            print(f"Hidden layers: {hidden_layers}")
            print(f"Dropout: {dropout_rate}")
            print(f"Model parameters: {self.model.count_params():,}")
            if use_order:
                order_dim = getattr(self, "order_dim", 32)
                print(f"Order dim: {order_dim}")
                print(f"DAG edges: {len(self._dag_edges)}")
                if use_dual:
                    fv_hidden = getattr(self, "fv_hidden", 32)
                    print(f"FV encoder hidden: {fv_hidden}")
                    print(f"FV encoder dropout: {float(getattr(self, 'fv_dropout', 0.0))}")
                    print(f"Gated fusion: {bool(getattr(self, 'gated_fusion', False))}")

    def _make_optimizer(self):
        """Resolve the Keras optimizer used to compile every model variant.

        Back-compatible by default: when neither LEARNING_RATE nor LR_SCHEDULE is
        present in config, this returns the optimizer *string* (e.g. "adam")
        exactly as before, so existing runs reproduce bit-for-bit. An explicit
        optimizer object is constructed only when a learning rate or schedule is
        requested — the tunables for probing whether the dual encoder's fast
        (~half-epoch) convergence is early-stopping before it exploits PseKRAAC.
        """
        import keras

        lr = getattr(self, "learning_rate", None)
        schedule = getattr(self, "lr_schedule", None)
        if lr is None and not schedule:
            return self.optimizer  # unchanged path — string like "adam"

        lr_value = float(lr) if lr is not None else 1e-3  # Keras Adam default

        if schedule and str(schedule).lower() == "cosine":
            # decay_steps needs the post-split training size; prepare_data() has
            # already populated self.X by the time build_model() compiles.
            val_split = float(getattr(self, "validation_split", 0.2))
            n_train = max(1, int(self.X.shape[0] * (1.0 - val_split)))
            bs = int(getattr(self, "batch_size", 32))
            if n_train > 10000:
                bs = min(bs, 64)  # mirror train_model's adaptive batch rule
            steps_per_epoch = max(1, -(-n_train // bs))  # ceil division
            decay_steps = steps_per_epoch * int(getattr(self, "epochs", 100))
            lr_value = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr_value, decay_steps=decay_steps
            )

        opt_name = str(getattr(self, "optimizer", "adam")).lower()
        if opt_name == "adam":
            return keras.optimizers.Adam(learning_rate=lr_value)
        # Fallback for any other named optimizer.
        opt = keras.optimizers.get(opt_name)
        opt.learning_rate = lr_value
        return opt

    def _build_flat_model(
        self, input_dim, num_classes, hidden_layers, dropout_rate,
        class_weights_tensor, tf,
    ):
        """Build the standard flat sigmoid model."""
        layer_list = [layers.Input(shape=(input_dim,))]
        for units in hidden_layers:
            layer_list.append(layers.Dense(units, activation="relu"))
            layer_list.append(layers.Dropout(dropout_rate))
        layer_list.append(layers.Dense(num_classes, activation="sigmoid"))

        model = models.Sequential(layer_list)

        def weighted_binary_crossentropy(y_true, y_pred):
            per_class_bce = -(
                y_true * tf.math.log(y_pred + 1e-7)
                + (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
            )
            weighted = per_class_bce * class_weights_tensor
            return tf.reduce_mean(weighted, axis=-1)

        model.compile(
            optimizer=self._make_optimizer(),
            loss=weighted_binary_crossentropy,
            metrics=["accuracy"],
        )

        self.model = model
        self._order_layer = None
        self._dag_edges = np.zeros((0, 2), dtype=np.int32)

    def _build_order_model(
        self, input_dim, num_classes, hidden_layers, dropout_rate,
        class_weights_tensor, tf,
    ):
        """Build the order embedding model with an order-violation loss."""
        if not _ORDER_AVAILABLE:
            raise ImportError(
                "Order embedding modules not found. Ensure "
                "protcast.model.order_embeddings and "
                "protcast.preprocessing.go_dag_edges are importable."
            )

        order_dim = getattr(self, "order_dim", 32)
        temperature = getattr(self, "order_temperature", 10.0)
        order_weight = getattr(self, "order_weight", 0.1)
        # ORDER_VARIANT: "soft" (softplus violation, default) or "hard" (ReLU)
        order_variant = str(getattr(self, "order_variant", "soft")).lower()
        soft_beta = float(getattr(self, "order_beta", 5.0))

        model, order_layer = build_order_embedding_model(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            order_dim=order_dim,
            temperature=temperature,
        )

        self._order_layer = order_layer

        # Extract DAG edges for the order-violation loss
        if self.go_dag is not None:
            self._dag_edges = extract_dag_edges(
                self.go_dag, self.go_ids, self.go_encoder
            )
        else:
            self._dag_edges = np.zeros((0, 2), dtype=np.int32)

        dag_edges_tensor = tf.constant(self._dag_edges, dtype=tf.int32)
        ow = tf.constant(order_weight, dtype=tf.float32)

        # Pick the order-violation loss variant once, outside the per-batch closure
        if order_variant == "soft":
            def _o_loss():
                return order_violation_loss_soft(
                    order_layer, dag_edges_tensor, beta=soft_beta
                )
        elif order_variant == "hard":
            def _o_loss():
                return order_violation_loss(order_layer, dag_edges_tensor)
        else:
            raise ValueError(
                f"Unknown ORDER_VARIANT '{order_variant}'. Use 'soft' or 'hard'."
            )

        def order_combined_loss(y_true, y_pred):
            # Weighted BCE (same as flat model)
            per_class_bce = -(
                y_true * tf.math.log(y_pred + 1e-7)
                + (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
            )
            weighted_bce = tf.reduce_mean(
                per_class_bce * class_weights_tensor, axis=-1
            )

            # Order-violation regularization (variant-dependent)
            return weighted_bce + ow * _o_loss()

        model.compile(
            optimizer=self._make_optimizer(),
            loss=order_combined_loss,
            metrics=["accuracy"],
        )

        self.model = model

        if self.verbose and len(self._dag_edges) > 0:
            extra = f", beta={soft_beta}" if order_variant == "soft" else ""
            print(
                f"Order-violation loss ({order_variant}): {len(self._dag_edges)} DAG edges, "
                f"weight={order_weight}{extra}"
            )

    def _build_order_model_dual(
        self, esm_dim, fv_dim, num_classes, hidden_layers, dropout_rate,
        class_weights_tensor, tf,
    ):
        """Build the dual-encoder order model.

        The concatenated input matrix self.X (shape N × (esm_dim + fv_dim)) is
        split at the esm_dim boundary inside the model using Lambda slicing, so
        the rest of the training pipeline (train_test_split, FmaxEarlyStopping,
        model.fit) requires no changes — they all see a single matrix.
        """
        if not _ORDER_AVAILABLE:
            raise ImportError(
                "Order embedding modules not found. Ensure "
                "protcast.model.order_embeddings is importable."
            )

        order_dim     = getattr(self, "order_dim", 32)
        temperature   = getattr(self, "order_temperature", 10.0)
        order_weight  = getattr(self, "order_weight", 0.1)
        order_variant = str(getattr(self, "order_variant", "soft")).lower()
        soft_beta     = float(getattr(self, "order_beta", 5.0))
        fv_hidden     = int(getattr(self, "fv_hidden", 32))
        fv_dropout    = float(getattr(self, "fv_dropout", 0.0))
        use_gated     = bool(getattr(self, "gated_fusion", False))

        # ── Build a wrapper model that splits its single input internally ──
        # This keeps the training loop identical to the single-encoder path:
        # model.fit(X, y) where X is the full concatenated matrix.
        combined_input = layers.Input(shape=(esm_dim + fv_dim,), name="combined_input")
        esm_slice = layers.Lambda(
            lambda z: z[:, :esm_dim], name="esm_slice"
        )(combined_input)
        fv_slice = layers.Lambda(
            lambda z: z[:, esm_dim:], name="fv_slice"
        )(combined_input)

        # ESM-C branch
        x = esm_slice
        for units in hidden_layers:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(dropout_rate)(x)

        # PseKRAAC branch — dedicated small MLP
        fv = layers.Dense(fv_hidden, activation="relu", name="fv_encoder")(fv_slice)
        # Optional regularisation on the FV branch. By default the ESM branch is
        # dropped at 0.5 while the FV branch passes un-dropped — a stable signal
        # that fits fast and can trip early-stopping before the model has learned
        # to exploit it. FV_DROPOUT lets us regularise this branch symmetrically.
        if fv_dropout > 0.0:
            fv = layers.Dropout(fv_dropout, name="fv_dropout")(fv)

        # Merge and project into order space. With GATED_FUSION the ESM branch
        # learns a per-dimension gate in (0, 1) over the FV features, so the
        # model can dial PseKRAAC up where it complements ESM (e.g. L6) and toward
        # zero where it is redundant/noisy (saturated deep levels), instead of the
        # fixed fv_hidden/(hidden[-1]+fv_hidden) weight a plain concatenation
        # imposes identically at every level.
        if use_gated:
            gate = layers.Dense(fv_hidden, activation="sigmoid", name="fv_gate")(x)
            fv = layers.Multiply(name="fv_gated")([fv, gate])

        # Merge and project into order space
        merged = layers.Concatenate(name="feature_merge")([x, fv])
        merged = layers.Dense(
            order_dim, activation="softplus", name="order_projection"
        )(merged)

        order_layer = OrderEmbeddingLayer(
            num_classes=num_classes,
            order_dim=order_dim,
            temperature=temperature,
            name="order_embeddings",
        )
        scores = order_layer(merged)

        import keras as _keras
        model = _keras.Model(
            inputs=combined_input, outputs=scores,
            name="order_embedding_model_dual",
        )
        self._order_layer = order_layer

        # DAG edges
        if self.go_dag is not None:
            self._dag_edges = extract_dag_edges(
                self.go_dag, self.go_ids, self.go_encoder
            )
        else:
            self._dag_edges = np.zeros((0, 2), dtype=np.int32)

        dag_edges_tensor = tf.constant(self._dag_edges, dtype=tf.int32)
        ow = tf.constant(order_weight, dtype=tf.float32)

        if order_variant == "soft":
            def _o_loss():
                return order_violation_loss_soft(
                    order_layer, dag_edges_tensor, beta=soft_beta
                )
        elif order_variant == "hard":
            def _o_loss():
                return order_violation_loss(order_layer, dag_edges_tensor)
        else:
            raise ValueError(
                f"Unknown ORDER_VARIANT '{order_variant}'. Use 'soft' or 'hard'."
            )

        def order_combined_loss(y_true, y_pred):
            per_class_bce = -(
                y_true * tf.math.log(y_pred + 1e-7)
                + (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
            )
            weighted_bce = tf.reduce_mean(
                per_class_bce * class_weights_tensor, axis=-1
            )
            return weighted_bce + ow * _o_loss()

        model.compile(
            optimizer=self._make_optimizer(),
            loss=order_combined_loss,
            metrics=["accuracy"],
        )
        self.model = model

        if self.verbose and len(self._dag_edges) > 0:
            extra = f", beta={soft_beta}" if order_variant == "soft" else ""
            print(
                f"Dual-encoder order-violation loss ({order_variant}): "
                f"{len(self._dag_edges)} DAG edges, weight={order_weight}{extra}"
            )

    def train_model(self) -> History:
        """Train the model with early stopping based on validation Fmax.

        Uses FmaxEarlyStopping to compute CAFA-style Fmax at the end of
        each epoch and stop when it plateaus. This trains directly toward
        the evaluation metric instead of using val_loss as a proxy.
        """
        if self.verbose:
            print("Starting training...")
            print(f"Data shapes - X: {self.X.shape}, y: {self.y.shape}")

        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y,
            test_size=self.validation_split,  # type: ignore
            random_state=self.random_state,
        )

        # Standardize on the training fold only (no validation leakage),
        # then apply the same transform to validation. Gated so legacy
        # callers are unaffected.
        if self.scale_features:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train).astype(np.float32)
            X_val = self.scaler.transform(X_val).astype(np.float32)
            if self.verbose:
                print("Applied StandardScaler (fit on train fold only)")

        self.X_train = X_train
        self.X_val = X_val
        self.y_val = y_val

        if self.verbose:
            print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

        # Early stopping on Fmax (the actual CAFA evaluation metric)
        fmax_callback = FmaxEarlyStopping(
            X_val=X_val,
            y_val=y_val,
            patience=self.patience,  # type: ignore
            min_delta=float(getattr(self, "min_delta", 0.001)),
            verbose=self.verbose,
        )

        # Also checkpoint on val_loss as a safety net
        checkpoint = ModelCheckpoint(
            filepath=f"{self.get_name()}.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        )

        callbacks = [fmax_callback, checkpoint]

        if self.use_tensorboard:
            log_dir = "logs/fit/" + time.strftime(
                "%m-%d-%Y-%H-%M-%S", time.localtime()
            )
            callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))

        train_start_time = time.time()

        adaptive_batch_size = self.batch_size  # type: ignore
        if X_train.shape[0] > 10000:
            adaptive_batch_size = min(adaptive_batch_size, 64)

        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,  # type: ignore
            batch_size=adaptive_batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose="auto",
            shuffle=True,
        )

        self.training_time = time.time() - train_start_time
        self.history = history

        # Best threshold comes from the callback (already restored best weights)
        self.best_threshold = fmax_callback.best_threshold
        self.best_fmax = fmax_callback.best_fmax

        if self.verbose:
            print(f"Training completed in {self.training_time:.2f}s")
            print(
                f"Best Fmax: {fmax_callback.best_fmax:.4f} "
                f"(epoch {fmax_callback.best_epoch + 1}, "
                f"threshold={fmax_callback.best_threshold:.2f})"
            )

        import gc
        gc.collect()

        return history  # type: ignore

    def compute_depth_metrics(self, y_true, y_pred):
        """Compute Fmax broken down by GO term depth in the DAG.

        Groups GO terms by their depth (longest path to root) and computes
        Fmax for each depth level. Enables apples-to-apples comparison with
        KNNClassifier on rare/specific terms.

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
        train_counts = self.X_train.shape[0] if hasattr(self, "X_train") else self.X.shape[0]
        label_counts = self.y.sum(axis=0)

        for depth, term_indices in sorted(depth_groups.items()):
            y_true_subset = y_true[:, term_indices]
            y_pred_subset = y_pred[:, term_indices]

            if y_true_subset.sum() == 0:
                continue

            fmax, threshold = calculate_fmax(y_true_subset, y_pred_subset)
            avg_count = float(label_counts[term_indices].mean())

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
        label_counts = self.y.sum(axis=0)

        buckets = {
            "rare_lt50": [],
            "medium_50_500": [],
            "common_gt500": [],
        }

        for i in range(len(self.go_ids)):
            count = label_counts[i]
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
            avg_count = float(label_counts[term_indices].mean())

            results[bucket_name] = {
                "fmax": fmax,
                "threshold": threshold,
                "n_terms": len(term_indices),
                "avg_train_count": avg_count,
            }

        return results

    @typechecked
    def save_model(self) -> None:
        """Save the trained model and threshold metadata."""
        self.model.save(f"{self.get_name()}.keras")

    @typechecked
    def log_model(self) -> None:
        """Log model, metrics, and artifacts to MLflow."""
        log_start_time = time.time()
        print("\n--- Starting MLflow Logging ---")

        mlflow = self._mlflow
        if mlflow is None:
            if self.verbose:
                print("mlflow not available; skipping logging")
            return

        try:
            from mlflow.tracking import MlflowClient
            from mlflow.entities import Metric
            from mlflow.models.signature import infer_signature
            from protcast.utils.mlflow_utils import save_run_metadata
        except Exception as e:
            if self.verbose:
                print("mlflow sub-imports failed; skipping logging:", e)
            return

        # Log parameters
        print("  > Logging parameters...", end="", flush=True)
        variant = "order" if getattr(self, "use_order_embeddings", False) else "flat"
        model_type = f"multi_label_{variant}"
        mlflow.log_params(self.params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("input_source", "esm_embeddings")
        mlflow.log_param("num_classes", len(self.go_ids))
        mlflow.log_param("feature_vector_length", self.vector_length)
        mlflow.log_param("best_threshold", round(self.best_threshold, 3))
        print(" done.")

        # Log dataset metadata
        print("  > Logging dataset metadata...", end="", flush=True)
        total_samples = self.X.shape[0]
        train_samples = self.X_train.shape[0]
        val_samples = total_samples - train_samples
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

        # Log Fmax and Smin (CAFA metrics)
        print("  > Computing and logging CAFA metrics...", end="", flush=True)
        y_val_pred = self.model.predict(self.X_val, verbose=0)
        fmax, _ = calculate_fmax(self.y_val, y_val_pred)
        mlflow.log_metric("val_fmax", round(fmax, 4))
        mlflow.log_metric("best_threshold", round(self.best_threshold, 4))

        from protcast.model.stats.utils import calculate_smin
        smin, smin_threshold = calculate_smin(self.y_val, y_val_pred)
        mlflow.log_metric("val_smin", round(smin, 4))
        mlflow.log_metric("smin_threshold", round(smin_threshold, 4))
        print(" done.")

        # Log depth-level metrics
        depth_metrics = self.compute_depth_metrics(self.y_val, y_val_pred)
        if depth_metrics:
            print("  > Logging depth-level metrics...", end="", flush=True)
            for depth, metrics in depth_metrics.items():
                mlflow.log_metric(f"fmax_depth_{depth}", round(metrics["fmax"], 4))
                mlflow.log_metric(f"n_terms_depth_{depth}", metrics["n_terms"])
                mlflow.log_metric(f"avg_count_depth_{depth}", round(metrics["avg_train_count"], 1))
            print(" done.")

        # Log frequency-bucket metrics
        freq_metrics = self.compute_frequency_metrics(self.y_val, y_val_pred)
        if freq_metrics:
            print("  > Logging frequency-bucket metrics...", end="", flush=True)
            for bucket, metrics in freq_metrics.items():
                mlflow.log_metric(f"fmax_{bucket}", round(metrics["fmax"], 4))
                mlflow.log_metric(f"n_terms_{bucket}", metrics["n_terms"])
            print(" done.")

        # Log epoch metrics
        print("  > Logging epoch metrics...", end="", flush=True)
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        history = self.history.history
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

        # Log model
        print("  > Uploading model...", end="", flush=True)
        signature = infer_signature(
            self.X_train, self.model.predict(self.X_train, verbose=0)
        )
        registered_name = self.params.get(
            "REGISTERED_MODEL_NAME", "multilabel_classifier.v0"
        )
        mlflow.keras.log_model(
            self.model,
            artifact_path="model",
            signature=signature,
            registered_model_name=registered_name,
        )
        print(" done.")

        mlflow.set_tag("Training Info", "MultiLabelClassifier full logging")
        mlflow.set_tag("model_type", model_type)

        self.logging_time = time.time() - log_start_time
        mlflow.log_metric("training_time_seconds", round(self.training_time, 2))
        mlflow.log_metric("total_logging_time_seconds", round(self.logging_time, 2))

        save_run_metadata(
            model_name=self.get_name(),
            run_id=run_id,
            experiment_name=self.params.get(
                "EXPERIMENT_NAME", "Default Experiment"
            ),
        )

        # End the MLflow run
        mlflow.end_run()
        print("--- MLflow Logging Complete ---")

    @classmethod
    def load_model(cls, model_path: Path) -> keras.models.Model:
        """Load a trained model from disk."""
        model = keras.models.load_model(model_path)
        return model  # type: ignore[return]

    @typechecked
    def get_name(self) -> str:
        """Generate model name using id."""
        return f"{self.id}_multilabel"


def get_confidence_label(score, threshold):
    """Map a sigmoid score to a human-readable confidence label,
    calibrated to the model's own Fmax-optimized threshold.

    Divides the range [threshold, 1.0] into four equal bands:

        Below threshold  →  not predicted (filtered out by decode_multilabel)
        Bottom quartile  →  LOW       — just above the decision boundary
        Second quartile  →  MEDIUM    — moderate confidence
        Third quartile   →  HIGH      — strong signal
        Top quartile     →  VERY_HIGH — near-certain prediction

    Parameters
    ----------
    score : float
        Sigmoid output for a single GO term (0.0 to 1.0).
    threshold : float
        The Fmax-optimized decision threshold from training.

    Returns
    -------
    str
        One of "VERY_HIGH", "HIGH", "MEDIUM", "LOW", or "BELOW_THRESHOLD".

    Examples
    --------
    >>> get_confidence_label(0.53, threshold=0.22)  # range 0.78, quartile=0.195
    'MEDIUM'
    >>> get_confidence_label(0.95, threshold=0.22)
    'VERY_HIGH'
    >>> get_confidence_label(0.10, threshold=0.22)
    'BELOW_THRESHOLD'
    """
    if score < threshold:
        return "BELOW_THRESHOLD"

    range_above = 1.0 - threshold
    quartile = range_above / 4.0

    if score >= threshold + 3 * quartile:
        return "VERY_HIGH"
    elif score >= threshold + 2 * quartile:
        return "HIGH"
    elif score >= threshold + quartile:
        return "MEDIUM"
    else:
        return "LOW"


class GOEncoder:
    """GOEncoder for multi-label classification.

    Maps GO IDs to integer indices for multi-hot encoding.
    Supports decoding sigmoid outputs back to GO term predictions.

    Example usage:
        go_encoder = GOEncoder('test')
        go_encoder.fit(['GO:0001', 'GO:0002', 'GO:0003'])

        # Encode a set of GO terms to multi-hot vector
        label = go_encoder.encode_multilabel({'GO:0001', 'GO:0003'})
        # → array([1., 0., 1.])

        # Decode sigmoid predictions above threshold
        predictions = np.array([0.9, 0.1, 0.8])
        terms = go_encoder.decode_multilabel(predictions, threshold=0.5)
        # → [('GO:0001', 0.9), ('GO:0003', 0.8)]
    """

    def __init__(self, id):
        self.id = id
        self.go_to_int = None
        self.int_to_go = None
        self.num_classes = 0

    def fit(self, go_ids):
        """Map GO IDs to integer indices."""
        unique_go_ids = sorted(set(go_ids))
        self.go_to_int = {go: i for i, go in enumerate(unique_go_ids)}
        self.int_to_go = {i: go for go, i in self.go_to_int.items()}
        self.num_classes = len(unique_go_ids)

    def encode(self, go_id):
        """Encode a single GO ID to its integer index."""
        if not self.go_to_int:
            raise ValueError("GOEncoder has not been fit to any GO IDs.")
        return self.go_to_int[go_id]

    def encode_multilabel(self, go_id_set):
        """Encode a set of GO IDs to a multi-hot vector.

        Parameters
        ----------
        go_id_set : set or list of str
            GO IDs annotated to a protein.

        Returns
        -------
        np.ndarray
            Multi-hot vector of shape (num_classes,).
        """
        if not self.go_to_int:
            raise ValueError("GOEncoder has not been fit to any GO IDs.")
        label = np.zeros(self.num_classes, dtype=np.float32)
        for go_id in go_id_set:
            if go_id in self.go_to_int:
                label[self.go_to_int[go_id]] = 1.0
        return label

    def decode_multilabel(self, probabilities, threshold=0.5):
        """Decode sigmoid probabilities to GO term predictions.

        Parameters
        ----------
        probabilities : np.ndarray
            Sigmoid output of shape (num_classes,) for a single protein.
        threshold : float
            Minimum probability to predict a GO term.

        Returns
        -------
        list of (str, float)
            List of (GO_ID, probability) tuples, sorted by probability descending.
        """
        if not self.int_to_go:
            raise ValueError("GOEncoder has not been fit to any GO IDs.")
        results = []
        for i, prob in enumerate(probabilities):
            if prob >= threshold:
                results.append((self.int_to_go[i], float(prob)))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def decode_probabilities(self, probabilities, top_k=1):
        """Decode probability arrays to top-k GO IDs (batch mode).

        Compatible with the original GOEncoder interface for inference scripts.

        Parameters
        ----------
        probabilities : np.ndarray
            Shape (num_samples, num_classes).
        top_k : int
            Number of top predictions to return per sample.

        Returns
        -------
        list of list of (str, float)
        """
        if not self.int_to_go:
            raise ValueError("GOEncoder has not been fit to any GO IDs.")
        top_indices = np.argsort(probabilities, axis=1)[:, -top_k:]
        result = []
        for i, indices in enumerate(top_indices):
            go_probs = [
                (self.int_to_go[idx], probabilities[i, idx]) for idx in indices
            ]
            result.append(sorted(go_probs, key=lambda x: x[1], reverse=True))
        return result

    def save(self):
        """Serialize to pickle file."""
        filename = f"{self.id}_GOEncoder.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load from pickle file."""
        with open(filename, "rb") as f:
            encoder = pickle.load(f)
        return encoder

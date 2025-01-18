from datetime import datetime
import os
import tensorflow as tf
import keras
import pandas as pd
import pickle
import numpy as np
import time
from pathlib import Path
from typeguard import typechecked

# from keras.utils import FeatureSpace, to_categorical  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.layers import Normalization, Dense  # type: ignore
from keras import layers, models
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore


from protcast.model.feature_vector import (
    get_ifeatpro_features,
    get_ifeatureomega_features,
)
from protcast.model.stats.utils import calculate_sensitivity_specificity
from protcast.model.stats.utils import calculate_f1_score

os.environ["KERAS_BACKEND"] = "tensorflow"


@typechecked
class MultiClassifier:
    """MultiClassifier
    This class ....

    Attributes
    ----------


    Methods
    -------
    init:
        Initialize
    make_featurespace:
        ...
    make_model:
        ...
    test_model:
        Test model against validation set, save results
        to a *tsv file.
    save_model:
        Save Keras model to file
    load_model:
        Load model from a file
    """

    @typechecked
    def __init__(
        self,
        algorithm: str,
        feature_creator: str,
        verbose: bool,
        save: bool,
        proteins: dict,
        optimizer: str = "adam",
        loss: str = "categorical_crossentropy",
        metrics: list = ["accuracy"],
        epochs: int = 100,
        batch_size: int = 32,
        fraction: float = 0.2,
        neurons: int = 32,
        dropout: float = 0.5,
        activation: str = "softmax",
        pred_threshold: float = 75.0,
        validation_split: float = 0.2,
        patience: int = 10,
    ) -> None:
        """__init__

        Parameters
        ----------
        algorithm : str
            _description_
        feature_creator: str
            Package that creates the feature vectors
        verbose: bool
            Verbosity
        save: bool
            Save the model, or not
        proteins: dict(dict)
            Primary key: GO id, secondary key: protein id, value: sequence
        vector_length: int
            Length of feature vector
        optimizer : str
            Optional, by default "adam"
        loss : str
            Optional, by default "categorical_crossentropy"
        metrics : list
            Optional, by default ["accuracy"]
        epochs : int
            By default 20
        fraction : float
            By default 0.2
        neurons : int
            By default 32
        dropout : float
            By default 0.5
        pred_threshold : float
            Probability threshold for classification, by default 80.0
        training_model: keras.src.engine.functional.Functional
            Trained model
        """
        self.algorithm = algorithm
        self.feature_creator = feature_creator
        self.verbose = verbose
        self.save = save
        self.proteins = proteins
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.fraction = fraction
        self.neurons = neurons
        self.dropout = dropout
        self.validation_split = validation_split
        self.pred_threshold = pred_threshold
        self.patience = patience
        self.go_ids = list()
        self.pids = list()
        self.features = list()
        self.go_encoder = GOEncoder()

    @typechecked
    def run(self) -> None:
        """run"""
        self.start_time = time.time()
        self.get_feature_vectors()
        self.prepare_data()
        self.build_model()
        self.train_model()

    @typechecked
    def get_feature_vectors(self) -> None:
        """get_feature_vectors
        Create feature vectors and create vectors as a list of lists,
        protein ids as a list of lists, and GO ids as a
        list
        """
        for go_id in self.proteins.keys():
            if self.feature_creator == "ifeatpro":
                features, pids = get_ifeatpro_features(
                    self.algorithm, self.proteins[go_id]
                )
            elif self.feature_creator == "iFeatureOmega":
                features, pids = get_ifeatureomega_features(
                    self.algorithm, self.proteins[go_id]
                )
            self.features.append(features)
            self.pids.append(pids)
            self.go_ids.append(go_id)
        self.vector_length = len(self.features[0][0])

    @typechecked
    def prepare_data(self) -> None:
        """prepare_data"""
        total_samples = sum(len(feature_set) for feature_set in self.features)
        X = np.zeros((total_samples, self.vector_length))

        # Create y (labels)
        y = np.zeros(total_samples, dtype=int)

        # Fill X and y with the actual data
        start_idx = 0
        for i, feature_set in enumerate(self.features):
            end_idx = start_idx + len(feature_set)
            X[start_idx:end_idx] = feature_set
            y[start_idx:end_idx] = i
            start_idx = end_idx

        # Convert y to categorical
        y_categorical = keras.utils.to_categorical(
            y, num_classes=len(self.go_ids)
        )
        self.X = X
        self.y = y_categorical

        if self.verbose:
            print(f"Shape of self.X: {self.X.shape}")
            print(f"Shape of self.y: {self.y.shape}")

    def build_model(self):
        """build_model

        The model architecture is a simple feed-forward neural network.
        The final layer uses num_classes for the number of units, ensuring it matches the number of GO ids.
        We're using 'softmax' activation in the final layer, which is appropriate for multi-class classification.
        The model is compiled with 'categorical_crossentropy' loss, which is suitable for multi-class problems with mutually exclusive classes.
        We're using 'accuracy' as a metric, but you might want to consider additional metrics like F1-score or area under the ROC curve, depending on your specific requirements.

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

        Softmax Activation:

        Used in the output layer for multi-class classification.
        Converts the raw output scores into probabilities that sum to 1.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j in all classes
        Advantages:
        - Provides a probability distribution over all classes.
        - Suitable for mutually exclusive classes.
        """
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(len(self.go_ids), activation="softmax"),
            ]
        )

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

        self.model = model

    """
    More flexible:

    def build_model(self, hidden_layers=[(128, 0.5), (64, 0.3)], optimizer='adam'):
        input_shape = (self.X.shape[1],)
        num_classes = len(self.go_ids)

        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))

        for units, dropout_rate in hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    """

    def train_model(self):
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=self.validation_split, stratify=self.y
        )
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )
        model_checkpoint = ModelCheckpoint(
            "best_model.h5", monitor="val_loss", save_best_only=True
        )
        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1,
        )
        # Load the best model
        self.model.load_weights("best_model.h5")
        return history

    @typechecked
    def save_model(self) -> None:
        """save_model"""
        self.model.save(f"{self.name}_{self.algorithm}.keras")

    @typechecked
    def load_model(self, model_path: Path) -> None:
        """load_model

        Parameters
        ----------
        model_path: Path
            Path to saved model
        """
        self.model = keras.models.load_model(model_path)


"""
        # Profiler callback
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        self.training_model.fit(
            train_tfds,
            epochs=self.epochs,
            validation_data=val_tfds,
            callbacks=[tensorboard_callback],
        )

    @typechecked
    def test_model(self) -> None:
        y_true = list()
        y_pred = list()
        f = open(f"{self.name}_{self.algorithm}.tsv", "w")
        for i, r in self.val_dataframe.iterrows():
            # Pre-process the sample you want a prediction from
            if r["target"] == 1.0:
                type = self.name
                y_true.append(1)
            else:
                type = f"non-{self.name}"
                y_true.append(0)
            del r["target"]
            sample_tfds = self.sample_preprocessing(r)
            # Get a prediction
            predictions = self.training_model.predict(sample_tfds)
            prob = float(100 * predictions[0][0])
            if prob >= self.pred_threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
            f.write(f"{type}\t{self.all_ids[i]}\t{prob}\n")
        sens, spec = calculate_sensitivity_specificity(y_true, y_pred)
        f.write(f"Sensitivity\t{sens}\tSpecificity\t{spec}\n")
        f.write(f"F1 score\t{calculate_f1_score(y_true, y_pred)}\n")
        f.write(
            f"Elapsed time\t{int(time.time() - self.start_time)} seconds\n"
        )
        f.write(f"Vector length\t{self.vector_length}")

"""


class GOEncoder:
    """GOEncoder

    # Example usage:
    go_encoder = GOEncoder()

    # Fit the encoder
    go_ids = ['GO:1224', 'GO:5678', 'GO:9101', 'GO:1224', 'GO:5678']
    go_encoder.fit(go_ids)

    # Encode GO IDs
    encoded = go_encoder.encode(['GO:1224', 'GO:5678', 'GO:9101'])
    print("Encoded:")
    print(encoded)

    # Save the encoder
    go_encoder.save('go_encoder.pkl')

    # Load the encoder
    loaded_encoder = GOEncoder.load('go_encoder.pkl')

    # Decode using the loaded encoder
    decoded = loaded_encoder.decode(encoded)
    print("Decoded (using loaded encoder):")
    print(decoded)

    # Decode probabilities
    probabilities = np.array([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
    top_go_ids = loaded_encoder.decode_probabilities(probabilities, top_k=2)
    print("Top 2 GO IDs with probabilities (using loaded encoder):")
    for go_probs in top_go_ids:
        print(go_probs)
    """

    def __init__(self):
        self.go_to_int = {}
        self.int_to_go = {}
        self.num_categories = 0

    def fit(self, go_ids):
        """fit
        Fit the encoder to a list of GO IDs.
        enumerate() creates the integers in *go_to_int*
        """
        unique_go_ids = sorted(set(go_ids))
        self.go_to_int = {go: i for i, go in enumerate(unique_go_ids)}
        self.int_to_go = {i: go for go, i in self.go_to_int.items()}
        self.num_categories = len(unique_go_ids)

    def encode(self, go_ids):
        """Encode a list of GO IDs to categorical."""
        if not self.go_to_int:
            raise ValueError("Encoder has not been fit to any GO IDs.")
        integer_encoded = [self.go_to_int[go] for go in go_ids]
        return keras.utils.to_categorical(
            integer_encoded, num_classes=self.num_categories
        )

    def decode(self, categorical):
        """Decode categorical back to GO IDs."""
        if not self.int_to_go:
            raise ValueError("Encoder has not been fit to any GO IDs.")
        integer_encoded = np.argmax(categorical, axis=1)
        return [self.int_to_go[i] for i in integer_encoded]

    def decode_probabilities(self, probabilities, top_k=1):
        """Decode probability distributions to top k GO IDs with their probabilities."""
        if not self.int_to_go:
            raise ValueError("Encoder has not been fit to any GO IDs.")
        top_indices = np.argsort(probabilities, axis=1)[:, -top_k:]
        result = []
        for i, indices in enumerate(top_indices):
            go_probs = [
                (self.int_to_go[idx], probabilities[i, idx]) for idx in indices
            ]
            result.append(sorted(go_probs, key=lambda x: x[1], reverse=True))
        return result

    def save(self, filename):
        """Serialize the GOEncoder to a file."""
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "go_to_int": self.go_to_int,
                    "int_to_go": self.int_to_go,
                    "num_categories": self.num_categories,
                },
                f,
            )

    @classmethod
    def load(cls, filename):
        """Deserialize a GOEncoder from a file."""
        with open(filename, "rb") as f:
            data = pickle.load(f)
        encoder = cls()
        encoder.go_to_int = data["go_to_int"]
        encoder.int_to_go = data["int_to_go"]
        encoder.num_categories = data["num_categories"]
        return encoder

import os
import keras
import pickle
import numpy as np
import time
from pathlib import Path
from typeguard import typechecked
import tensorflow as tf
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
    get_features_vectors:
        ...
    prepare_data:
        ...
    build_model:
        ...
    train_model:
        ...
    test_modeL:
        Test model against validation set, save results
        to a *tsv file
    save_model:
        Save Keras model to file
    load_model:
        Class method to load model from a file
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
        pred_threshold: float = 75.0,
        validation_split: float = 0.2,
        patience: int = 10,
    ) -> None:
        """__init__

        Parameters
        ----------
        algorithm : str
            Name of algorithm creating the feature vector
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
        training_model:
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
        self.fraction = fraction
        self.neurons = neurons
        self.dropout = dropout
        self.validation_split = validation_split
        self.pred_threshold = pred_threshold
        self.patience = patience
        self.go_ids = list()
        self.pids = list()
        self.features = list()
        # self.go_encoder = GOEncoder()

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
        num_samples = sum(len(go_set) for go_set in self.features)
        X = np.zeros((num_samples, self.vector_length))
        y = np.zeros(num_samples, dtype=int)

        go_encoder = GOEncoder()
        go_encoder.fit(self.go_ids)

        # Need to account for different number of proteins for different GO ids
        start_idx = 0
        for i, (go_id, go_set) in enumerate(zip(self.go_ids, self.features)):
            end_idx = start_idx + len(go_set)
            X[start_idx:end_idx] = go_set
            y[start_idx:end_idx] = go_encoder.encode(go_id)
            start_idx = end_idx

        self.X = X
        # Convert integers into a binary class matrix
        self.y = keras.utils.to_categorical(y, num_classes=len(self.go_ids))

        if self.verbose:
            print(f"Shape of self.X: {self.X.shape}")
            print(f"Shape of self.y: {self.y.shape}")
        """
        Shape of self.X: (305, 100)
        Shape of self.y: (305, 3)
        
        Vector length: 100, number of proteins (samples): 305, number of GO ids (classes): 3
        """
        go_encoder.save()

    def build_model(self):
        """build_model

        The final layer uses num_classes for the number of units, ensuring it matches the number of GO ids.
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
            # The minimum change in the monitored quantity to qualify as an improvement
            min_delta=0.001,
            # Number of epochs with no improvement after which training will be stopped
            patience=self.patience,
            restore_best_weights=True,
        )
        # val_loss measures the error on the validation set
        loss_checkpoint = ModelCheckpoint(
            f"{self.get_name()}.h5",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        )
        # acc_checkpoint = ModelCheckpoint(
        #     "best_model_accuracy.h5",
        #     monitor="val_accuracy",
        #     mode="max",
        #     save_best_only=True,
        #     verbose=1,
        # )
        log_dir = "logs/fit/" + time.strftime(
            "%m-%d-%Y-%H-%M-%S", time.localtime()
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                early_stopping,
                loss_checkpoint,
                tensorboard_callback,
            ],
            verbose=1,
        )

        return history

    @typechecked
    def save_model(self) -> None:
        """save_model"""
        self.model.save(f"{self.get_name()}.keras")

    @classmethod
    def load_model(
        cls, model_path: Path
    ) -> keras.src.models.sequential.Sequential:
        """load_model

        Parameters
        ----------
        model_path: Path
            Path to saved model
        """
        model = keras.models.load_model(model_path)
        return model

    @typechecked
    def get_name(self) -> str:
        return f"{time.strftime('%m-%d-%Y-%H-%M-%S', time.localtime())}_{self.algorithm}"


"""

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
    go_encoder.save()

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
        self.go_to_int = dict()
        self.int_to_go = dict()
        self.num_categories = 0

    def fit(self, go_ids):
        """fit
        Map the list of GO IDs to a list of integers, and integers to GO ids.
        """
        unique_go_ids = sorted(set(go_ids))
        self.go_to_int = {go: i for i, go in enumerate(unique_go_ids)}
        self.int_to_go = {i: go for go, i in self.go_to_int.items()}
        self.num_categories = len(unique_go_ids)

    def encode(self, go_id):
        """Encode a GO ID to a category number"""
        if not self.go_to_int:
            raise ValueError("Encoder has not been fit to any GO IDs.")
        return self.go_to_int[go_id]

    def decode(self, categorical):
        """Decode categories back to GO IDs."""
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

    def save(self):
        """Serialize the GOEncoder to a file."""
        filename = f"{time.strftime('%m-%d-%Y-%H-%M-%S', time.localtime())}_GOEncoder.pickle"
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

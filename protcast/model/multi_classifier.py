import os
import keras
import pickle
import numpy as np
import time
from pathlib import Path
from typeguard import typechecked
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.callbacks import TensorBoard  # type: ignore
from sklearn.model_selection import train_test_split
from protein_feature_vectors import Calculator

# from protcast.model.stats.utils import calculate_sensitivity_specificity
# from protcast.model.stats.utils import calculate_f1_score

os.environ["KERAS_BACKEND"] = "tensorflow"


@typechecked
class MultiClassifier:
    """MultiClassifier
    This class uses feature vectors to train and test a multi-class classifier.
    It uses the Keras Sequential API.

    Attributes
    ----------


    Methods
    -------
    init:
        Initialize
    run:
      Run the methdds that create the model and train it
    get_features_vectors:
        Get feature vectors using ifeatpro or iFeatureOmega
    prepare_data:
        Encode the GO ids and make numpy data structures
    build_model:
        Define the model layers and parameters
    train_model:
        Add data to model and train
    save_model:
        Save Keras model to *hf file
    load_model:
        Class method to load model from a file
    get_name:
        Make a model name with timestamp
    """

    @typechecked
    def __init__(
        self,
        algorithm: str,
        verbose: bool,
        proteins: dict,
        optimizer: str = "adam",
        loss: str = "categorical_crossentropy",
        metrics: list = ["accuracy"],
        epochs: int = 100,
        batch_size: int = 32,
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
        verbose: bool
            Verbosity
        proteins: dict(dict)
            Primary key: GO id, secondary key: protein id, value: sequence
        vector_length: int
            Length of feature vector
        optimizer : str
            Default "adam"
        loss : str
            Default "categorical_crossentropy"
        metrics : list
            Default ["accuracy"]
        epochs : int
            By default 20
        neurons : int
            By default 32
        dropout : float
            By default 0.5
        pred_threshold : float
            Probability threshold for classification, by default 80.0
        """
        self.algorithm = algorithm
        self.verbose = verbose
        self.proteins = proteins
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.neurons = neurons
        self.dropout = dropout
        self.validation_split = validation_split
        self.pred_threshold = pred_threshold
        self.patience = patience
        self.go_ids = list()
        self.pids = list()
        self.features = list()

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
        Create feature vectors and get vectors as a list of lists (per GO id),
        protein ids as a list of lists (per GO id), and GO ids as a
        list
        """
        fv = Calculator()
        for go_id in self.proteins.keys():
            fv.get_feature_vectors(self.algorithm, pdict=self.proteins[go_id])
            # encodings is a pandas DataFrame
            pids = [x[0] for x in fv.encodings.iterrows()]
            self.pids.append(pids)
            vals = [x[1].tolist() for x in fv.encodings.iterrows()]
            self.features.append(vals)
            self.go_ids.append(go_id)
        # Arbitrary choice, get its length
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
        # Convert integers into a binary matrix
        self.y = to_categorical(y, num_classes=len(self.go_ids))

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
        We're using 'accuracy' as a metric, but consider metrics like F1-score or area under the ROC curve.

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
            filepath=f"{self.get_name()}.keras",
            # "best_model.h5",
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
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
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
    def load_model(cls, model_path: Path) -> keras.models.Sequential:
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


class GOEncoder:
    """GOEncoder

    # Example usage:
    go_encoder = GOEncoder()
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

    def __init__(self):
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
            return self.int_to_go.get(categorical, None)
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
        """Serialize the GOEncoder to a file."""
        filename = f"{time.strftime('%m-%d-%Y-%H-%M-%S', time.localtime())}_GOEncoder.pickle"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Deserialize a GOEncoder from a file."""
        with open(filename, "rb") as f:
            encoder = pickle.load(f)
        return encoder

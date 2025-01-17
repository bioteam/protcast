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
from keras import layers

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
        epochs: int = 20,
        batch_size: int = 32,
        fraction: float = 0.2,
        neurons: int = 32,
        dropout: float = 0.5,
        activation: str = "softmax",
        pred_threshold: float = 75.0,
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
        self.pred_threshold = pred_threshold
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
        """prepare_data
        Set up the size and type (float) of the FeatureSpace object and add the
        column names starting with 1

        features = dict()
        for count in range(len(self.target_features[0])):
            features[str(count)] = FeatureSpace.float_normalized()
            self.column_names.append(str(count))
        self.feature_space = FeatureSpace(features=features)

        # Add "target" column name
        self.column_names.append("target")

        """
        # First, let's determine the number of samples and feature length
        # First, let's determine the total number of samples and feature length
        num_go_ids = len(self.go_ids)
        total_samples = sum(len(feature_set) for feature_set in self.features)
        feature_length = len(
            self.features[0][0]
        )  # Assuming all feature vectors have the same length

        # Create X with the correct shape
        X = np.zeros((total_samples, feature_length))

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
        y_cat = keras.utils.to_categorical(y, num_classes=num_go_ids)

        self.X = X
        self.y = y_cat

        if self.verbose:
            print(f"Shape of self.X: {self.X.shape}")
            print(f"Shape of self.y: {self.y.shape}")

    def build_model(self):
        """build_model"""
        x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(
            self.feature_layer
        )
        x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(x)
        """
       
        """
        outputs = layers.TimeDistributed(
            layers.Dense(3, activation="softmax")
        )(x)

        self.model = keras.Model(inputs=self.input_layer, outputs=outputs)

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train_model(self):
        self.model.fit(
            self.X,
            self.y_categorical,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
        )

        """
        # Create the FeatureSpace dynamically
        feature_columns = {
            f"feature_{i}": X[:, i] for i in range(self.vector_length)
        }
        self.feature_space = FeatureSpace(
            features=feature_columns,
            crosses=None,
            output_mode="concat",
        )
        # Apply normalization after creating FeatureSpace
        normalized_features = tf.keras.layers.Normalization()(
            self.feature_space(feature_columns)
        )
        self.feature_space.adapt(X)


    def make_model(self):
       
        self.model = Sequential(
            [
                self.feature_space,
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(len(self.go_ids), activation=self.activation),
            ]
        )

        # Compile the model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

        # Train the model
        self.model.fit(
            self.input_data,
            self.y_categorical,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )

"""

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

    @typechecked
    def prepare_data(self) -> tuple:
        all_dataframe = pd.DataFrame(
            self.all_features, columns=self.column_names
        )
        self.val_dataframe = all_dataframe.sample(
            frac=self.fraction, random_state=1337
        )
        # The index holds the row names, don't need them for training
        self.train_dataframe = all_dataframe.drop(self.val_dataframe.index)
        train_tfds = self.dataframe_to_tfdataset(self.train_dataframe)
        val_tfds = self.dataframe_to_tfdataset(self.val_dataframe)

        # why batched into 32?
        train_tfds = train_tfds.batch(32)
        val_tfds = val_tfds.batch(32)

        # The function adapt() that adapts the Featurespace to the training data only works on
        # datasets dicts of feature values so we have to make a version of the dataset with the labels stripped
        train_tfds_no_labels = train_tfds.map(lambda x, _: x)
        # train_ds_with_no_labels = [x for x, _ in train_ds]

        # adapt() is kind of magical. During this time the FeatureSpace will:
        # Index the set of possible values for the categorical features, compute mean and variance to aid with
        # normalizing the numerical features plus compute the value boundaries for the different bins for
        # numerical features to discretize.
        self.feature_space.adapt(train_tfds_no_labels)

        # Attempt at asynch preprocessing not sure if CLAB hardware is optimized for this yet though
        # Running it as part of the tf.data pipeline instead of the model itself
        processed_train_tfds = train_tfds.map(
            lambda x, y: (self.feature_space(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        processed_train_tfds = processed_train_tfds.prefetch(tf.data.AUTOTUNE)

        processed_val_tfds = val_tfds.map(
            lambda x, y: (self.feature_space(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        processed_val_tfds = processed_val_tfds.prefetch(tf.data.AUTOTUNE)

        return processed_train_tfds, processed_val_tfds

    @typechecked
    def make_model(
        self, train_tfds: tf.data.Dataset, val_tfds: tf.data.Dataset
    ) -> None:
        # The first layer is the encoded features as a KerasTensor
        encoded_features = self.feature_space.get_encoded_features()
        # Create a dense layer with 32 neurons and apply the ReLU activation function for non-linearity.
        kt = keras.layers.Dense(self.neurons, activation="relu")(
            encoded_features
        )
        # Apply a dropout layer with a rate of 0.5 to the input data represented by kt.
        # Dropout() is a regularization technique commonly used to prevent overfitting.
        kt = keras.layers.Dropout(self.dropout)(kt)
        # Create a dense layer with a single neuron and apply the sigmoid activation function
        # which outputs 0 or 1. This is a common approach for the output layer in binary classification.
        ktoutput = keras.layers.Dense(1, activation="sigmoid")(kt)
        # Create a keras.src.engine.functional.Functional object
        self.training_model = keras.Model(
            inputs=encoded_features,
            outputs=ktoutput,
        )
        #
        self.training_model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )

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

    @typechecked
    def dataframe_to_tfdataset(
        self, dataframe: pd.DataFrame
    ) -> tf.data.Dataset:
        # The original dataframe passed to method is unchanged
        dataframe = dataframe.copy()
        labels = dataframe.pop("target")
        tfds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        tfds = tfds.shuffle(buffer_size=len(dataframe))
        return tfds

    @typechecked
    def sample_preprocessing(
        self, sample: pd.core.series.Series
    ) -> tf.data.Dataset:
        # Convert pandas Series into dataframe
        sample_frame = pd.DataFrame([sample])
        # Convert datafrane into Tensorflow Datasest with stub target
        sample_tfds = tf.data.Dataset.from_tensor_slices(
            (dict(sample_frame), [0])
        )
        # Batch of 1 since there's only 1 sample
        sample_tfds = sample_tfds.batch(1)
        # Pre-process the dataset using the FeatureSpace map
        processed_sample_tfds = sample_tfds.map(
            lambda x, y: (self.feature_space(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return processed_sample_tfds

"""

""" 
1. One-Hot Encoding (OHE): This is a common preprocessing step that converts your categorical features into numerical representations.

from keras.utils import to_categorical

# Suppose you have a feature 'animal' with categories ['cat', 'dog', 'human']

X = pd.DataFrame({'animal': ['cat', 'dog', 'cat', 'dog']})

X_onehot = to_categorical(X['animal'])
print(X_onehot)
Output:

array([[1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.]])

2. Embeddings: This is a more modern and efficient technique to represent categorical features in numerical form.

from keras.layers import Embedding

# Define your embedding layer with a specified vocabulary size (e.g., number of categories)
embedding_layer = Embedding(input_dim=3, output_dim=10)

# Process your input data using the embedding layer
X_embedded = embedding_layer(X_onehot)

3. Multi-Class Classification: Now that you've converted your categorical features into numerical representations, you can train a Keras model for multi-class classification.

from keras.models import Sequential

# Define your model architecture
model = Sequential()
model.add(embedding_layer)  # Use the embedding layer to process input data
model.add(Dense(64, activation='relu'))  # Hidden layer with ReLU activation
model.add(Dense(3, activation='softmax'))  # Output layer with softmax activation for multi-class classification

# Compile your model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train your model on your dataset
model.fit(X_embedded, Y_onehot, epochs=10)        

"""
"""
from tensorflow import keras
from keras_feature_space import FeatureSpace

# Assuming 'X_train', 'y_train', 'X_test', 'y_test' are your data

fs = FeatureSpace(
    model=keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')  # Multi-class output
    ]),
    loss='categorical_crossentropy', # done
    metrics=['accuracy'] # done
)

fs.fit(X_train, y_train)
y_pred = fs.predict(X_test)
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
        """Fit the encoder to a list of GO IDs."""
        unique_go_ids = sorted(set(go_ids))
        self.go_to_int = {go: i for i, go in enumerate(unique_go_ids)}
        self.int_to_go = {i: go for go, i in self.go_to_int.items()}
        self.num_categories = len(unique_go_ids)

    def encode(self, go_ids):
        """Encode a list of GO IDs to categorical."""
        if not self.go_to_int:
            raise ValueError("Encoder has not been fit to any GO IDs yet.")
        integer_encoded = [self.go_to_int[go] for go in go_ids]
        return keras.utils.to_categorical(
            integer_encoded, num_classes=self.num_categories
        )

    def decode(self, categorical):
        """Decode categorical back to GO IDs."""
        if not self.int_to_go:
            raise ValueError("Encoder has not been fit to any GO IDs yet.")
        integer_encoded = np.argmax(categorical, axis=1)
        return [self.int_to_go[i] for i in integer_encoded]

    def decode_probabilities(self, probabilities, top_k=1):
        """Decode probability distributions to top k GO IDs with their probabilities."""
        if not self.int_to_go:
            raise ValueError("Encoder has not been fit to any GO IDs yet.")
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

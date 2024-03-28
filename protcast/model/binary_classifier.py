import os
import tensorflow as tf
import keras
import pandas as pd
from typeguard import typechecked
from keras.utils import FeatureSpace
from protcast.model.feature_vector import get_ifeatpro_features

os.environ["KERAS_BACKEND"] = "tensorflow"


@typechecked
class BinaryClassifier:
    """BinaryClassifier
    This class ....

    Attributes
    ----------


    Methods
    -------
    init:
        Initialize
    make_featurespace:
        ...
    prepare_data:
        ...
    make_model:
        ...
    test_model:
        Test model against validation set, save results
        to a *tsv file.
    save_model:
        Save Keras model to file
    """

    @typechecked
    def __init__(
        self,
        name: str,
        target_seqs: dict,
        non_target_seqs: dict,
        algorithm: str,
        optimizer: str = "adam",
        loss: str = "binary_crossentropy",
        metrics: list = ["accuracy"],
        epochs: int = 20,
        fraction: float = 0.2,
        neurons: int = 32,
        dropout: float = 0.5
    ) -> None:
        self.name = name
        self.target_seqs = target_seqs
        self.non_target_seqs = non_target_seqs
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.fraction = fraction
        self.neurons = neurons
        self.dropout = dropout

    @typechecked
    def run(self) -> None:
        self.make_featurespace()
        train_dataset, val_dataset = self.prepare_data()
        self.make_model(train_dataset, val_dataset)

    @typechecked
    def make_featurespace(self) -> None:
        # Get feature vectors for all proteins as a list of lists
        target_features, target_ids = get_ifeatpro_features(
            self.algorithm, self.target_seqs
        )
        non_target_features, non_target_ids = get_ifeatpro_features(
            self.algorithm, self.non_target_seqs
        )
        # Set up the size and type (float) in the FeatureSpace object and get the column names
        features = dict()
        self.column_names = list()
        for count in range(len(target_features[0])):
            features[str(count)] = FeatureSpace.float_normalized()
            self.column_names.append(str(count))
        self.feature_space = FeatureSpace(features=features)

        # Add target values of 0 or 1 to data
        target_features = [x + [1] for x in target_features]
        non_target_features = [x + [0] for x in non_target_features]

        # Add "target" column name to data
        self.column_names.append("target")

        self.all_features = target_features + non_target_features
        self.all_ids = target_ids + non_target_ids

    @typechecked
    def prepare_data(self) -> tuple:
        all_dataframe = pd.DataFrame(self.all_features, columns=self.column_names)
        self.val_dataframe = all_dataframe.sample(frac=self.fraction, random_state=1337)
        self.train_dataframe = all_dataframe.drop(self.val_dataframe.index)
        train_ds = self.dataframe_to_tfdataset(self.train_dataframe)
        val_ds = self.dataframe_to_tfdataset(self.val_dataframe)

        # why batched into 32?
        train_ds = train_ds.batch(32)
        val_ds = val_ds.batch(32)

        # The function adapt() that adapts the Featurespace to the training data only works on
        # datasets dicts of feature values so we have to make a version of the dataset with the labels stripped
        train_ds_with_no_labels = train_ds.map(lambda x, _: x)
        # train_ds_with_no_labels = [x for x, _ in train_ds]

        # adapt() is kind of magical. During this time the FeatureSpace will:
        # Index the set of possible values for the categorical features, compute mean and variance to aid with
        # normalizing the numerical features plus compute the value boundaries for the different bins for
        # numerical features to discretize.
        self.feature_space.adapt(train_ds_with_no_labels)

        # Attempt at asynch preprocessing not sure if CLAB hardware is optimized for this yet though
        # Running it as part of the tf.data pipeline instead of the model itself
        preprocessed_train_ds = train_ds.map(
            lambda x, y: (self.feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

        preprocessed_val_ds = val_ds.map(
            lambda x, y: (self.feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

        return preprocessed_train_ds, preprocessed_val_ds

    @typechecked
    def make_model(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> None:
        encoded_features = self.feature_space.get_encoded_features()
        # Create a dense layer with 32 neurons and apply the ReLU activation function to
        # the data received from encoded_features.
        x = keras.layers.Dense(self.neurons, activation="relu")(encoded_features)
        # Apply a dropout layer with a rate of 0.5 to the input data represented by x.
        # Dropout() is a regularization technique commonly used to prevent overfitting.
        x = keras.layers.Dropout(self.dropout)(x)
        # Create a dense layer with a single neuron and apply the sigmoid activation function
        # to its input. This is a common approach for the output layer in binary classification.
        output = keras.layers.Dense(1, activation="sigmoid")(x)

        self.training_model = keras.Model(
            inputs=encoded_features,
            outputs=output,
        )
        self.training_model.compile(
            optimizer=self.optimizer, 
            loss=self.loss, 
            metrics=self.metrics
        )
        # Here's a pipeline model that will be trained and called seperately
        self.training_model.fit(
            train_ds,
            epochs=self.epochs,
            validation_data=val_ds,
        )

    @typechecked
    def test_model(self):
        with open(f"{self.name}_{self.algorithm}.tsv", "w") as f:
            for i, r in self.val_dataframe.iterrows():
                if r["target"] == 1.0:
                    type = self.name
                else:
                    type = f"non-{self.name}"
                # Pre-process the sample you want a prediction from
                del r["target"]
                sample_ds = self.sample_preprocessing(r)
                # Get a prediction
                predictions = self.training_model.predict(sample_ds)
                f.write(f"{type}\t{self.all_ids[i]}\t{100 * predictions[0][0]:.2f}\n")

    @typechecked
    def dataframe_to_tfdataset(self, dataframe: pd.DataFrame) -> tf.data.Dataset:
        # The original dataframe passed to method is unchanged
        dataframe = dataframe.copy()
        labels = dataframe.pop("target")
        tfds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        tfds = tfds.shuffle(buffer_size=len(dataframe))
        return tfds

    @typechecked
    def sample_preprocessing(self, sample: pd.core.series.Series) -> tf.data.Dataset:
        # Convert pandas Series into dataframe
        sample_frame = pd.DataFrame([sample])
        # Convert datafrane into Tensorflow Datasest with stub target
        sample_ds = tf.data.Dataset.from_tensor_slices((dict(sample_frame), [0]))
        # Batch of 1 since there's only 1 sample
        sample_ds = sample_ds.batch(1)
        # Pre-process the dataset using the FeatureSpace map
        preprocessed_sample_ds = sample_ds.map(
            lambda x, y: (self.feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        return preprocessed_sample_ds

    @typechecked
    def save_model(self) -> None:
        self.training_model.save(f"{self.name}_{self.algorithm}.keras")

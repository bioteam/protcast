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
        self.column_names = list()

    @typechecked
    def run(self) -> None:
        self.get_feature_vectors()
        self.make_featurespace()
        train_tfdataset, val_tfdataset = self.prepare_data()
        self.make_model(train_tfdataset, val_tfdataset)

    @typechecked
    def get_feature_vectors(self) -> None:
        # Get feature vectors for all proteins as a list of lists
        self.target_features, target_ids = get_ifeatpro_features(
            self.algorithm, self.target_seqs
        )
        self.non_target_features, non_target_ids = get_ifeatpro_features(
            self.algorithm, self.non_target_seqs
        )
        self.all_ids = target_ids + non_target_ids

    @typechecked
    def make_featurespace(self) -> None:
        # Set up the size and type (float) in the FeatureSpace object and get the column names
        features = dict()
        for count in range(len(self.target_features[0])):
            features[str(count)] = FeatureSpace.float_normalized()
            self.column_names.append(str(count))
        self.feature_space = FeatureSpace(features=features)

        # Add "target" column name
        self.column_names.append("target")

        # Add target values of 0 or 1 to data
        self.target_features = [x + [1] for x in self.target_features]
        self.non_target_features = [x + [0] for x in self.non_target_features]
        self.all_features = self.target_features + self.non_target_features


    @typechecked
    def prepare_data(self) -> tuple:
        all_dataframe = pd.DataFrame(self.all_features, columns=self.column_names)
        self.val_dataframe = all_dataframe.sample(frac=self.fraction, random_state=1337)
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
            lambda x, y: (self.feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        processed_train_tfds = processed_train_tfds.prefetch(tf.data.AUTOTUNE)

        processed_val_tfds = val_tfds.map(
            lambda x, y: (self.feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        processed_val_tfds = processed_val_tfds.prefetch(tf.data.AUTOTUNE)

        return processed_train_tfds, processed_val_tfds

    @typechecked
    def make_model(self, train_tfds: tf.data.Dataset, val_tfds: tf.data.Dataset) -> None:
        """make_model
        
        """
        # The first layer is the encoded features as a KerasTensor
        encoded_features = self.feature_space.get_encoded_features()
        # Create a dense layer with 32 neurons and apply the ReLU activation function to
        # introduce non-linearity.
        kt = keras.layers.Dense(self.neurons, activation="relu")(encoded_features)
        # Apply a dropout layer with a rate of 0.5 to the input data represented by x.
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
            optimizer=self.optimizer, 
            loss=self.loss, 
            metrics=self.metrics
        )
        # 
        self.training_model.fit(
            train_tfds,
            epochs=self.epochs,
            validation_data=val_tfds,
        )

    @typechecked
    def test_model(self) -> None:
        with open(f"{self.name}_{self.algorithm}.tsv", "w") as f:
            for i, r in self.val_dataframe.iterrows():
                if r["target"] == 1.0:
                    type = self.name
                else:
                    type = f"non-{self.name}"
                # Pre-process the sample you want a prediction from
                del r["target"]
                sample_tfds = self.sample_preprocessing(r)
                # Get a prediction
                predictions = self.training_model.predict(sample_tfds)
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
        sample_tfds = tf.data.Dataset.from_tensor_slices((dict(sample_frame), [0]))
        # Batch of 1 since there's only 1 sample
        sample_tfds = sample_tfds.batch(1)
        # Pre-process the dataset using the FeatureSpace map
        processed_sample_tfds = sample_tfds.map(
            lambda x, y: (self.feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        return processed_sample_tfds

    @typechecked
    def save_model(self) -> None:
        self.training_model.save(f"{self.name}_{self.algorithm}.keras")

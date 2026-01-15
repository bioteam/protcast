import os
import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace
import pytest

os.environ["KERAS_BACKEND"] = "tensorflow"

pytestmark = pytest.mark.integration


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def test_featurespace_demo_integration():
    """Integration demo that downloads a small CSV, adapts a FeatureSpace and trains a tiny model.
    This is marked integration and skipped in unit runs.
    """
    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"

    def prediction_preprocessing(sample_dict, feature_space):
        # Convert dict into dataframe
        sample_frame = pd.DataFrame([sample_dict])
        # Convert datafrane into Tensor Datasest with stub target
        sample_ds = tf.data.Dataset.from_tensor_slices((dict(sample_frame), [0]))
        # Batch of 1 since there's only 1 sample
        sample_ds = sample_ds.batch(1)
        # Pre-process the dataset using the FeatureSpace map
        preprocessed_sample_ds = sample_ds.map(
            lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        return preprocessed_sample_ds

    feature_space = FeatureSpace(
        features={
            # Categorical features encoded as integers
            "sex": FeatureSpace.integer_categorical(num_oov_indices=0),
            "cp": FeatureSpace.integer_categorical(num_oov_indices=0),
            "fbs": FeatureSpace.integer_categorical(num_oov_indices=0),
            "restecg": FeatureSpace.integer_categorical(num_oov_indices=0),
            "exang": FeatureSpace.integer_categorical(num_oov_indices=0),
            "ca": FeatureSpace.integer_categorical(num_oov_indices=0),
            # Categorical feature encoded as string
            "thal": FeatureSpace.string_categorical(num_oov_indices=0),
            # Numerical features to discretize
            "age": FeatureSpace.float_discretized(num_bins=30),
            # Numerical features to normalize
            "trestbps": FeatureSpace.float_normalized(),
            "chol": FeatureSpace.float_normalized(),
            "thalach": FeatureSpace.float_normalized(),
            "oldpeak": FeatureSpace.float_normalized(),
        "slope": FeatureSpace.float_normalized(),
    },
    # We create additional features by hashing
    # value co-occurrences for the
    # following groups of categorical features.
    crosses=[("sex", "age"), ("thal", "ca")],
    # The hashing space for these co-occurrences
    # wil be 32-dimensional.
    crossing_dim=32,
    # Our utility will one-hot encode all categorical
    # features and concat all features into a single
    # vector (one vector per sample).
    output_mode="concat",
    )

    sample = {
    "age": 60,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": "fixed",
}
    dataframe = pd.read_csv(file_url)
    val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
    train_dataframe = dataframe.drop(val_dataframe.index)
    train_ds = dataframe_to_dataset(train_dataframe)
    val_ds = dataframe_to_dataset(val_dataframe)

    # why batched into 32?
    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    train_ds_with_no_labels = train_ds.map(lambda x, _: x)

    feature_space.adapt(train_ds_with_no_labels)

    preprocessed_train_ds = train_ds.map(
        lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

    preprocessed_val_ds = val_ds.map(
        lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

    encoded_features = feature_space.get_encoded_features()

    # Create a dense layer with 32 neurons and applies the ReLU activation function to
    # the data received from encoded_features.
    x = keras.layers.Dense(32, activation="relu")(encoded_features)
    # Apply a dropout layer with a rate of 0.5 to the input data represented by x.
    # Dropout() is a regularization technique commonly used to prevent overfitting.
    x = keras.layers.Dropout(0.5)(x)
    # Create a dense layer with a single neuron and apply the sigmoid activation function
    # to its input. This is a common approach for the output layer in binary classification.
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    training_model = keras.Model(inputs=encoded_features, outputs=output)
    training_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    # Keep epochs small for demo
    training_model.fit(preprocessed_train_ds, epochs=1, validation_data=preprocessed_val_ds)

    # Pre-process the sample you want a prediction from
    preprocessed_sample_ds = prediction_preprocessing(sample)

    # Get a prediction
    predictions = training_model.predict(preprocessed_sample_ds)

    assert predictions.shape[0] >= 1

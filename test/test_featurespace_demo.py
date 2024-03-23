import os
import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace

os.environ["KERAS_BACKEND"] = "tensorflow"

file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def prediction_preprocessing(sample_dict):
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

# The function adapt() that adapts the Featurespace to the training data only works on
# datasets dicts of feature values so we have to make a version of the dataset with the labels stripped
train_ds_with_no_labels = train_ds.map(lambda x, _: x)

# adapt is kind of magical. During this time the FeatureSpace will:
# Index the set of possible values for the categorical features, compute mean and variance to aid with
# normalizing the numerical features plus compute the value boundaries for the different bins for
# numerical features to discretize.
feature_space.adapt(train_ds_with_no_labels)

# Attempt at asynch preprocessing not sure if CLAB hardware is optimized for this yet though
# Running it as part of the tf.data pipeline instead of the model itself
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

# Here's an end-to-end model just to demo the functionality
# dict_inputs = feature_space.get_inputs()
# inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)

training_model = keras.Model(inputs=encoded_features, outputs=output)
training_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# Here's a pipeline model that will be trained and called seperately
training_model.fit(
    preprocessed_train_ds,
    epochs=20,
    validation_data=preprocessed_val_ds,
    verbose=2,
)

# Pre-process the sample you want a prediction from
preprocessed_sample_ds = prediction_preprocessing(sample)

# Get a prediction
predictions = training_model.predict(preprocessed_sample_ds)

print(
    f"Sample has a {100 * predictions[0][0]:.2f}% probability "
    "of having a heart disease, as evaluated by our model."
)

training_model.save("example_model.keras")

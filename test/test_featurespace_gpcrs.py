import os
import sys
import tensorflow as tf
import keras
import pandas as pd
from pathlib import Path
from keras.utils import FeatureSpace
from Bio import SeqIO

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from protcast.model.feature_vector import get_ifeatpro_features  # noqa: E402

os.environ["KERAS_BACKEND"] = "tensorflow"


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


gpcr_seqs = SeqIO.to_dict(
    SeqIO.parse("test/data/uniprotkb_gpcrs.fasta", "fasta")
)
non_gpcr_seqs = SeqIO.to_dict(
    SeqIO.parse("test/data/uniprotkb_non-gpcrs.fasta", "fasta")
)

# Get feature vectors for all proteins as a list of lists
gpcr_features, gpcr_ids = get_ifeatpro_features("ctriad", gpcr_seqs)
non_gpcr_features, non_gpcr_ids = get_ifeatpro_features(
    "ctriad", non_gpcr_seqs
)

# Set up the size and type (float) in the FeatureSpace object and get the column names
features = dict()
column_names = list()
for count in range(len(gpcr_features[0])):
    features[str(count)] = FeatureSpace.float_normalized()
    column_names.append(str(count))
feature_space = FeatureSpace(features=features)

# Add target values of 0 or 1 and "target" column name
gpcr_features = [x + [1] for x in gpcr_features]
non_gpcr_features = [x + [0] for x in non_gpcr_features]
column_names.append("target")

all_features = gpcr_features + non_gpcr_features
all_ids = gpcr_ids + non_gpcr_ids

dataframe = pd.DataFrame(all_features, columns=column_names)
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
# train_ds_with_no_labels = [x for x, _ in train_ds]

# adapt() is kind of magical. During this time the FeatureSpace will:
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

# Create a dense layer with 32 neurons and apply the ReLU activation function to
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

training_model = keras.Model(
    inputs=encoded_features,
    outputs=output,
)
training_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)
# Here's a pipeline model that will be trained and called seperately
training_model.fit(
    preprocessed_train_ds,
    epochs=20,
    validation_data=preprocessed_val_ds,
)

for i, r in val_dataframe.iterrows():
    if r["target"] == 1.0:
        type = "GPCR"
    else:
        type = "Non-GPCR"
    # Pre-process the sample you want a prediction from
    del r["target"]
    preprocessed_sample_ds = prediction_preprocessing(r)
    # Get a prediction
    predictions = training_model.predict(preprocessed_sample_ds)
    print(f"{type}\t{all_ids[i]}\t{100 * predictions[0][0]:.2f}")

training_model.save("test_featurespace_gpcrs.keras")

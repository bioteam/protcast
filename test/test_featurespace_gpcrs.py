import tensorflow as tf
import keras
import pandas as pd
from keras.utils import FeatureSpace
from Bio import SeqIO
import pytest

pytestmark = pytest.mark.integration


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def prediction_preprocessing(sample_dict, feature_space):
    # Convert dict into dataframe
    sample_frame = pd.DataFrame([dict(sample_dict)])
    # Convert dataframe into Tensor Dataset with stub target
    sample_ds = tf.data.Dataset.from_tensor_slices((dict(sample_frame), [0]))
    sample_ds = sample_ds.batch(1)
    preprocessed_sample_ds = sample_ds.map(
        lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    return preprocessed_sample_ds


def test_featurespace_gpcrs_integration():
    """Integration demo that generates feature vectors from GPCR fixtures and trains a small model.
    This test is marked integration and will be skipped by default in unit runs.
    """
    # import inside test to avoid import-time side-effects
    from protein_feature_vectors import Calculator

    gpcr_seqs = SeqIO.to_dict(SeqIO.parse("test/data/uniprotkb_gpcrs.fasta", "fasta"))
    non_gpcr_seqs = SeqIO.to_dict(SeqIO.parse("test/data/uniprotkb_non-gpcrs.fasta", "fasta"))

    # Get feature vectors for all proteins as a list of lists
    fv = Calculator()
    # protein_feature_vectors.Calculator populates `encodings` on the instance;
    # extract features and ids from the DataFrame after calling.
    fv.get_feature_vectors("CTriad", pdict=gpcr_seqs)
    gpcr_enc = fv.encodings
    gpcr_ids = list(gpcr_enc.index)
    gpcr_features = gpcr_enc.values.tolist()

    fv.get_feature_vectors("CTriad", pdict=non_gpcr_seqs)
    non_gpcr_enc = fv.encodings
    non_gpcr_ids = list(non_gpcr_enc.index)
    non_gpcr_features = non_gpcr_enc.values.tolist()

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
    training_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Keep epochs small for demo
    training_model.fit(preprocessed_train_ds, epochs=1, validation_data=preprocessed_val_ds)

    # Pre-process one sample and predict (no assertions, this is a smoke integration test)
    sample = dict(val_dataframe.iloc[0])
    sample.pop("target", None)
    preprocessed_sample_ds = prediction_preprocessing(sample, feature_space)
    preds = training_model.predict(preprocessed_sample_ds)
    assert preds.shape[0] == 1

    # quick smoke save to ensure API works (file overwritten if exists)
    training_model.save("test_featurespace_gpcrs.keras")


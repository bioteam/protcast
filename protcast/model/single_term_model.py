import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace
from sklearn import metrics as sk_metrics
from tensorflow import keras
from tensorflow.keras import backend as K


class SingleTermModel(keras.callbacks.Callback):
    """SingleTerms
    This class ....

    Attributes
    ----------
    keras.callbacks.Callback : object
        ....
    dense1: Keras object
        First Dense NN layer
    dense2: Keras object
        Second Dense NN layer

    Methods
    -------
    init:
        Initialize
    on_train_begin:
        ...
    on_epoch_end:
        ...
    """

    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def on_train_begin(self, logs={}):
        """on_train_begin
        ...

        Parameters
        ----------
        logs: dict
            ...

        Returns
        -------
        None
        """
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        """on_epoch_end
        ...

        Parameters
        ----------
        epoch: int
            ...
        logs: dict
            ...

        Returns
        -------
        None
        """
        val_predict = (
            np.asarray(self.model.predict(self.model.validation_data[0]))
        ).round()
        val_target = self.model.validation_data[1]
        val_f1 = f1_score(val_target, val_predict)
        val_recall = recall_score(val_target, val_predict)
        val_precision = precision_score(val_target, val_predict)
        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precision)
        print(
            f" - val_f1: {val_f1} - val_precision: {val_precision} - val_recall: {val_recall}"
        )

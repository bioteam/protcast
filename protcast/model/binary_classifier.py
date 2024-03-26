import os
import sys
import tensorflow as tf
import keras
import pandas as pd
from pathlib import Path
from keras.utils import FeatureSpace

os.environ["KERAS_BACKEND"] = "tensorflow"


class BinaryClassifier():
    """BinaryClassifier
    This class ....

    Attributes
    ----------
    

    Methods
    -------
    init:
        Initialize
    """

    def __init__(self):
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(10, activation="softmax")


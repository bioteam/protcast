import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras import layers
from keras.models import Model
from keras.utils import FeatureSpace
from keras import backend as K

class MyModel(Model):
    def __init__(self, num_classes=2):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(32, activation='relu')
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# Instantiate the model
model = MyModel(num_classes=2)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

import unittest
import numpy as np
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import sys

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

# Why is this not importing? Also note objects imported in test_training straight up didn't inherit anything why?
# from protcast.model.simple_model import MyModel  # noqa: E402
from protcast.model.binary_classifier import BinaryClassifier  # noqa: E402

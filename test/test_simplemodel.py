import unittest
import numpy as np
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

# Why is this not importing? Also note objects imported in test_training straight up didn't inherit anything why?
# from protcast.model.simple_model import MyModel  # noqa: E402
from protcast.model.single_term_model import SingleTermModel  # noqa: E402


class TestMyModel(unittest.TestCase):
    def setUp(self):
        self.model = SingleTermModel(num_classes=2)
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Generate dummy training data
        self.x_train = np.random.random((1000, 20))
        self.y_train = to_categorical(
            np.random.randint(2, size=(1000, 1)), num_classes=2
        )

    def test_model_can_be_compiled(self):
        self.assertIsNotNone(self.model.optimizer)
        self.assertIsNotNone(self.model.loss)
        self.assertIsNotNone(self.model.compiled_metrics)

    def test_model_produces_output_of_expected_shape(self):
        predictions = self.model.predict(self.x_train)
        self.assertEqual(predictions.shape, (1000, 2))

    # This test is failing
    def test_model_can_be_fitted(self):
        history = self.model.fit(
            self.x_train, self.y_train, batch_size=32, epochs=1
        )
        print("******************")
        print(history)
        print("******************")
        self.assertIsNotNone(history)


if __name__ == "__main__":
    unittest.main()

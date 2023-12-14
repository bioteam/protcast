import numpy as np
from sklearn import metrics as sk_metrics
from tensorflow import keras
from tensorflow.keras import backend as K


class Metrics(keras.callbacks.Callback):
    """Metrics
    This class ....

    Attributes
    ----------
    keras.callbacks.Callback : object
        ....

    Methods
    -------
    on_train_begin:
        ...
    on_epoch_end:
        ...
    """

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
            np.asarray(
                self.model.predict(self.model.validation_data[0])
            )
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


class SklearnMetrics(Metrics):
    """SklearnMetrics
    This class ....

    Attributes
    ----------
    Metrics : object
        ....

    Methods
    -------
    f1_score:
        ...
    recall_score:
        ...
    precision_score:
        ...
    """

    def f1_score(y_true, y_pred):
        """f1_score
        ...

        Parameters
        ----------
        y_true: ...
            ...
        y_pred:
            ...

        Returns
        -------
        float
        """
        return sk_metrics.f1_score(y_true, y_pred)

    def recall_score(y_true, y_pred):
        """recall_score
        ...

        Parameters
        ----------
        y_true: float
            ...
        y_pred: float
            ...

        Returns
        -------
        None
        """
        return sk_metrics.recall_score(y_true, y_pred)

    def precision_score(y_true, y_pred):
        """precision_score
        ...

        Parameters
        ----------
        y_true: float
            ...
        y_pred: float
            ...

        Returns
        -------
        float
        """
        return sk_metrics.precision_score(y_true, y_pred)


class KerasMetrics(Metrics):
    """KerasMetrics
    This class ....

    Attributes
    ----------
    Metrics : object
        ....

    Methods
    -------
    recall_score:
        ...
    precision_score:
        ...
    f1_score:
        ...
    """

    def recall_score(y_true, y_pred):
        """recall_score
        ...

        Parameters
        ----------
        y_true: float
            ...
        y_pred: float
            ...

        Returns
        -------
        None
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_score(y_true, y_pred):
        """precision_score
        ...

        Parameters
        ----------
        y_true: float
            ...
        y_pred: float
            ...

        Returns
        -------
        float
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (
            predicted_positives + K.epsilon()
        )
        return precision

    def f1_score(y_true, y_pred):
        """f1_score
        ...

        Parameters
        ----------
        y_true: float
            ...
        y_pred: float
            ...

        Returns
        -------
        float
        """
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2 * (
            (precision * recall) / (precision + recall + K.epsilon())
        )

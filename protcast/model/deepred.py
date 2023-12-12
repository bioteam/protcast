from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from protcast import BP, CC, MF
from protcast.model.custom_metrics import KerasMetrics, SklearnMetrics

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from preprocessing.protcast_dataset import DeepredDataset


class Deepred:
    """Deepred
    This class represents the complete Protein Oracle model.
    It encapsulates all the submodels built from a DeepredDataset

    Attributes
    ----------
    model_dateset : DeepredDataset
        A list of submodels where each element is a list of the
        output classes (GO terms) of the submodel.

    Methods
    -------
    init:
        Initialize
    add_submodel:
        ...
    fit_submodel:
        ...
    build:
        ...
    load:
        ...
    evaluate:
        ...
    build_submodels_from_dataset:
        ...
    build_submodels_from_tree:
        ...
    summary:
        ...
    to_text:
        ...
    """

    def __init__(self, model_dataset: DeepredDataset):
        """__init__
        ...

        Parameters
        ----------
        model_dataset: DeepredDataset
            ...

        Returns
        -------
        None
        """
        self.dataset = model_dataset
        self.submodels_count = 0
        self.submodels = self.build_submodels_from_dataset()

    def add_submodel(self, submodel) -> None:
        """add_submodel
        Append a submodel to a list of submodels

        Parameters
        ----------
        submodel: DeepredDataset
            ...

        Returns
        -------
        None
        """
        self.submodels.append(submodel)

    def fit_submodel(self, namespace, level, bucket, index, x_hat, y_hat):
        """fit_submodel
        ...

        Parameters
        ----------
        namespace: str
            ...
        level: int
            ...
        bucket:
            ...
        index:
            ...
        x_hat: list of floats
            ...
        y_hat: list of floats
            ...

        Returns
        -------
        None

        """
        submodel = self.submodels[namespace][level][bucket][index]
        submodel.summary()
        submodel.model.fit(
            x_hat,
            y_hat,
            calbacks=[submodel.keras_metrics, submodel.sklearn_metrics],
        )

    def build():
        pass

    def load():
        pass

    def evaluate():
        pass

    def build_submodels_from_dataset(self):
        """build_submodels_from_dataset
        Builds a dict with 3 keys, CC, BP, and MF, where each
        value is a dict of dicts.

        Parameters
        ----------
        None

        Returns
        -------
        dict
        """
        submodels_trees = {}
        submodels_trees[BP] = self.build_submodels_from_tree(
            self.dataset.bp_submodels_tree, BP
        )
        submodels_trees[CC] = self.build_submodels_from_tree(
            self.dataset.cc_submodels_tree, CC
        )
        submodels_trees[MF] = self.build_submodels_from_tree(
            self.dataset.mf_submodels_tree, MF
        )
        return submodels_trees

    def build_submodels_from_tree(self, submodels_tree: dict, namespace):
        """build_submodels_from_tree
        Builds a submodel tree for one namespace, a nested dict where the primary
        key is a GO level and the secondary key is a 'bucket', an integer classifying
        GO terms based on the number of proteins annotated with that term.

        Parameters
        ----------
        submodels_tree: dict of dicts
            ...
        namespace: str
            ...

        Returns
        -------
        Submodel: dict of dicts
        """
        submodel_tree = {}
        for level, buckets in submodels_tree.items():
            submodel_tree[level] = {}
            for bucket, submodels in buckets.items():
                submodel_tree[level][bucket] = []
                for i, submodel in enumerate(submodels):
                    submodel = SubModel(
                        self.submodels_count,
                        namespace,
                        go_terms=[
                            (term.id, len(term.annotations)) for term in submodel
                        ],
                        level=level,
                        min_annotations=bucket,
                        model_index=i,
                        input_size=343,
                    )
                    self.submodels_count += 1
                    submodel_tree[level][bucket].append(submodel)
        return submodel_tree

    def summary(self):
        print("------------------")
        print("------------------")

    def to_text(self):
        """to_text
        Creates text version of submodels by GO namespace

        Parameters
        ----------
        None

        Returns
        -------
        str
        """
        submodels = []
        for submodel in self.submodels[BP]:
            submodels.append(submodel.to_text())
            submodels.append("\n")
        for submodel in self.submodels[CC]:
            submodels.append(submodel.to_text())
            submodels.append("\n")
        for submodel in self.submodels[MF]:
            submodels.append(submodel.to_text())
            submodels.append("\n")
        return "\n".join(submodels)


class SubModel:
    """SubModel
    This class builds the Dense layers of the submodel and complies the submodel.
        DEEPred submodels user the Relu activation function and use softmax
        in the last layer.

        Attributes
        ----------
        name: type
            ....

        Methods
        -------
        init:
            Initialize, build 2 Dense layers, and compile the submodel
        to_text:
            ...
    """

    def __init__(
        self,
        global_index,
        namespace: str,
        go_terms: list[(str, int)],
        level: int,
        min_annotations: int,
        model_index: int,
        input_size: int,
        neurons_l1: int = 1600,
        neurons_l2: int = 400,
        activation_func: str = "relu",
        dropout: float = 0.5,
    ) -> None:
        self.global_index = global_index
        self.name = f"{namespace}_{level}_{min_annotations}_{model_index}"
        self.namespace = namespace
        self.min_annotations = min_annotations
        self.go_terms, self.go_terms_annotations = go_terms
        self.num_classes = len(self.go_terms)
        self.level = level
        self.neurons_l1 = neurons_l1
        self.neurons_l2 = neurons_l2
        self.dropout = dropout
        self.activation_func = activation_func
        self.keras_metrics = KerasMetrics()
        self.sklearn_metrics = SklearnMetrics()

        inputs = keras.Input(shape=(input_size))
        dense_1 = layers.Dense(self.neurons_l1, activation=self.activation_func)(inputs)
        drop_1 = layers.Dropout(self.dropout)(dense_1)
        dense_2 = layers.Dense(self.neurons_l2, activation=self.activation_func)(drop_1)
        drop_2 = layers.Dropout(self.dropout)(dense_2)
        outputs = layers.Dense(self.num_classes, activation="sigmoid")(drop_2)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
        )

    def summary(self):
        self.model.summary()

    def to_text(self):
        """to_text
        Creates text version of submodels

        Parameters
        ----------
        None

        Returns
        -------
        str
        """
        submodel_str = []
        submodel_str.append(f"Index: {self.global_index}")
        submodel_str.append(f"Name: {self.names}")
        submodel_str.append(f"Namespace: {self.namespace}")
        submodel_str.append(f"Level: {self.level}")
        submodel_str.append(f"Index: {self.index}")
        submodel_str.append(f"# GO Terms: {len(self.go_terms)}")
        submodel_str.append("# GO Term\tAnnotations")
        for got, annots in zip(self.go_terms, self.go_terms_annotations):
            submodel_str.append(f"{got}\t{annots}")
        return "\n".join(submodel_str)

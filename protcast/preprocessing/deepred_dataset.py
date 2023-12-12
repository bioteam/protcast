from __future__ import annotations

import os

from datetime import datetime
import h5py
from imblearn.over_sampling._mlsmote import MLSMOTE
import logging as log
import numpy as np
from pathlib import Path
import pickle
import pprint
from typeguard import typechecked

from protcast import BP, CC, MF
from protcast.preprocessing import utils
from protcast.preprocessing.dataset import Dataset
from protcast.preprocessing.ontology import GOTerm


CTRIAD = "ctriad"
SUPPORTED_FEATURE_VECTORS = [
    CTRIAD,
]
FEATURE_VECTOR_LENGTHS = {
    CTRIAD: 343,
}
DEFAULT_BUCKETS = [30, 100, 300, 500]
DEFAULT_TRAINING_PARTITION = 0.6
DEFAULT_VALIDATION_PARTITION = 0.1
DEFAULT_TEST_PARTITION = 0.3
DEFAULT_FEATURE_VECTOR_NAME = CTRIAD
DEFAULT_GO_TERMS_PER_SUBMODEL = 4
DEFAULT_TARGET_IMBALANCE_RATIO = 3


@typechecked
class SubModelDataset:
    """SubModelDataset
    The SubModelDataset holds the training, validation and test data
    for a submodel as well as indexes related to the DeepredDataset.

    Attributes
    ----------
    name: type
        ....

    Methods
    -------
    init:
        Initialize
    write_dataset_to_files:
        ...
    """

    def __init__(
        self,
        go_terms: list[GOTerm],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_orig: np.ndarray,
        y_orig: np.ndarray,
        samples_pid: np.ndarray,
        feature_vector_name: str,
        global_index: int,
        namespace: str,
        ontology_level: int,
        bucket: int,
        level_index: int,
    ) -> None:
        """__init__
        ...

        Parameters
        ----------
        go_terms: list
            list of GOTerms
        X_train: numpy
            2D array where each row is the feature vector for the
        protein.
        y_train: numpy
            2D array where each row is a one-hot encoded vector
            indicating whether the protein is annotatated with the GO term.
        X_val: numpy
            x for validation
        y_val: numpy
            y for validation
        X_test: numpy
            x for test
        y_test:
            y for test
        X_orig: numpy
            x for original
        y_orig:
            y for original
        samples_pid: numpy
            Array of protein ids
        global_index: int
            ...
        ontology_level: int
            ...
        bucket: int
            ...
        level_index: int
            ...

        Returns
        -------
        None
        """
        self.go_terms = go_terms
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X_orig = X_orig
        self.y_orig = y_orig
        self.samples_pid = samples_pid
        self.feature_vector_name = feature_vector_name
        self.global_index: int = global_index
        self.namespace = namespace
        self.ontology_level: int = ontology_level
        self.bucket: int = bucket
        self.level_index: int = level_index

    @typechecked
    def write_dataset_to_files(self, output_dir: Path):
        """write_dataset_to_files
        Saves x_train, y_train, x_val, y_val, x_test, and y_test to disk

        Parameters
        ----------
        namespace: str
            GO namespace
        bucket: int
            ...
        output_dir: Path
            Directory for the 6 output files

        Returns
        -------
        None
        """
        np.savetxt(
            output_dir / f"{self.feature_vector_name}_model_{self.namespace}_"
            f"{self.ontology_level}_{self.bucket}_{self.level_index}_"
            "x_train.mat",
            self.X_train,
            fmt="%.3e",
        )

        np.savetxt(
            output_dir / f"{self.feature_vector_name}_model_{self.namespace}_"
            f"{self.ontology_level}_{self.bucket}_{self.level_index}_"
            "y_train.mat",
            self.y_train,
            fmt="%d",
        )

        np.savetxt(
            output_dir / f"{self.feature_vector_name}_model_{self.namespace}_"
            f"{self.ontology_level}_{self.bucket}_{self.level_index}_"
            "x_val.mat",
            self.X_val,
            fmt="%.3e",
        )

        np.savetxt(
            output_dir / f"{self.feature_vector_name}_model_{self.namespace}_"
            f"{self.ontology_level}_{self.bucket}_{self.level_index}_"
            "y_val.mat",
            self.y_val,
            fmt="%d",
        )

        np.savetxt(
            output_dir / f"{self.feature_vector_name}_model_{self.namespace}_"
            f"{self.ontology_level}_{self.bucket}_{self.level_index}_"
            "x_test.mat",
            self.X_test,
            fmt="%.3e",
        )

        np.savetxt(
            output_dir / f"{self.feature_vector_name}_model_{self.namespace}_"
            f"{self.ontology_level}_{self.bucket}_{self.level_index}_"
            "y_test.mat",
            self.y_test,
            fmt="%d",
        )

    def write_dataset_to_hdf5(self, hdf5_file):
        """write_dataset_to_hdf5
        Write submodel datasets to HD5 for one namespace

        Parameters
        ----------
        namespace: str
            GO namespace
        hdf5_file: str
            HDF5 content
        level: int
            GO level
        bucket: int
            dataset bucket
        submodel: object
            SubModelDataset

        Returns
        -------
        None
        """
        x_train_dataset_name = (
            f"model_{self.namespace}_{self.ontology_level}_{self.bucket}_"
            f"{self.level_index}_x_train"
        )
        hdf5_file.create_dataset(x_train_dataset_name, data=self.X_train)
        y_train_dataset_name = (
            f"model_{self.namespace}_{self.ontology_level}_{self.bucket}_"
            f"{self.level_index}_y_train"
        )
        hdf5_file.create_dataset(y_train_dataset_name, data=self.y_train)
        x_val_dataset_name = (
            f"model_{self.namespace}_{self.ontology_level}_{self.bucket}_"
            f"{self.level_index}_x_val"
        )
        hdf5_file.create_dataset(x_val_dataset_name, data=self.X_val)
        y_val_dataset_name = (
            f"model_{self.namespace}_{self.ontology_level}_{self.bucket}_"
            f"{self.level_index}_y_val"
        )
        hdf5_file.create_dataset(y_val_dataset_name, data=self.y_val)
        x_test_dataset_name = (
            f"model_{self.namespace}_{self.ontology_level}_{self.bucket}_"
            f"{self.level_index}_x_test"
        )
        hdf5_file.create_dataset(x_test_dataset_name, data=self.X_test)
        y_test_dataset_name = (
            f"model_{self.namespace}_{self.ontology_level}_{self.bucket}_"
            f"{self.level_index}_y_test"
        )
        hdf5_file.create_dataset(y_test_dataset_name, data=self.y_test)


class DeepredDataset:
    """DeepredDataset
    This class organizes all the GO terms and the submodels.
    The data structure to organize all the GO terms are:
    - {BP|CC|MF}_go_terms_tree: they are a dictionary for each GO namespace
    (i.e. biological process, cellular component, modelcular function)
    where:
    - Keys: the levels in the GO tree
    - Values: dictionaries where:
        - Keys: Represent the lower limit for the number of sequences
        associated with the GO term ("buckets")
            - a Bucket contains the GO terms associated with some range
              (number) of proteins
        - Values: List of GO terms.

    Graphically it looks like:
    {
        1: { # GO Level 1
            30: [x0, x1, x2, ...],
            # GO terms in level 1 associated with 30 to 100 protein seqs.
            100: [y0, y1, y2, ...],
                ...
        }
        2: { # GO Level 2
            30: [w0, w1, ...],
            100: [z0, z1, ...]
        },
        ...
    }

    The data structure to organize all the sample data (X, y) in different
    submodels are:
    - {BP|CC|MF}_submodels_tree: they are a dictionary for each GO
    namespace (i.e. biological process, cellular component, modelcular
    function) where:
    - Keys: Are the levels in the GO tree
    - Values: are dictionaries where:
        - Keys: Represent the lower limit for the number of sequences
        associated with the GO terms ("buckets")
        - Values: List of SubModelDataset

    Graphically it looks like:
    {
        1: { # GO Level
            30: [SubModelDataset(data1), SubModelDataset(data2)],
            # GO terms in level 1 associated with 30 to 100 protein seqs.
            100: [...]
            ...
            }
        2: { # GO Level
            30: [...],
            100: [...]
        },
        ...
    }

    Attributes
    ----------
    name: type
        ....

    Methods
    -------
    init:
        Initialize
        ...
    _filter_non_compatible_proteins:
        ...
    _build_go_terms_tree:
        ...
    _get_bucket:
        ...
    _build_submodels_tree:
        ...
    _build_submodel_dataset:
        ...
    _build_feature_vectors:
        ...
    write_datasets_to_files:
        ...
    _write_submodel_dataset_tree_to_files:
        ...
    write_datasets_to_hdf5:
        ...
    write_submodel_dataset_tree_to_hdf5:
        ...
    write_submodel_dataset_to_hdf5:
        ...
    from_serialized_protcast_dataset:
        ...
    from_files:
        ...
    save:
        ...
    summary:
        ...
    summarize_tree:
        ...
    print_submodels_tree:
        ...
    print_submodel:
        ...
    """

    @typechecked
    def __init__(
        self,
        dataset: Dataset,
        training_partition: float = DEFAULT_TRAINING_PARTITION,
        validation_partition: float = DEFAULT_VALIDATION_PARTITION,
        test_partition: float = DEFAULT_TEST_PARTITION,
        buckets: list[int] = DEFAULT_BUCKETS,
        go_terms_per_submodel: int = DEFAULT_GO_TERMS_PER_SUBMODEL,
        feature_vector_name: str = DEFAULT_FEATURE_VECTOR_NAME,
        target_imbalance_ratio: float | None = DEFAULT_TARGET_IMBALANCE_RATIO,
    ) -> None:
        """__init__
        Initialize and build feature vectors, and the GO term trees and
        SubmodelDataset trees, one per GO namespace

        Parameters
        ----------
        dataset: Dataset
            ...
        training_partition: float
            DEFAULT_TRAINING_PARTITION
        validation_partition: float
            DEFAULT_VALIDATION_PARTITION
        test_partition: float
            DEFAULT_TEST_PARTITION
        buckets: list[int]
            DEFAULT_BUCKETS
        go_terms_per_submodel: int
            DEFAULT_GO_TERMS_PER_SUBMODEL
        feature_vector_name: str
            DEFAULT_FEATURE_VECTOR_NAME
        target_imbalance_ratio: float
            DEFAULT_TARGET_IMBALANCE_RATIO,

        Returns
        -------
        None
        """
        partition_sum = (
            training_partition + validation_partition + test_partition
        )

        if partition_sum != 1.0:
            print(
                f"Parititons sum is {partition_sum}. They should add up to 1"
            )
            exit(1)

        if feature_vector_name not in SUPPORTED_FEATURE_VECTORS:
            print(
                f"Feature vector '{feature_vector_name}' is not currently "
                "supported. Supported feature vectors are: "
                f"{SUPPORTED_FEATURE_VECTORS}"
            )
            exit(1)

        self.created_at = datetime.now()
        self.dataset = dataset
        self.feature_vector_name = feature_vector_name
        self.buckets = buckets
        self.buckets.sort()
        self.go_terms_per_submodel = go_terms_per_submodel
        self.training_partition = training_partition
        self.validation_partition = validation_partition
        self.test_partition = test_partition
        self.target_imbalance_ratio = target_imbalance_ratio
        self.submodel_global_index = 0

        self._filter_non_compatible_proteins()

        log.info("Building feature vectors.")
        self.feature_vectors: dict[
            str, list[float]
        ] = self._build_feature_vectors()
        # Proteins to be included in model
        self.filtered_go_terms = {}

        log.info("Building GO term trees.")
        (
            self.bp_go_terms_tree,
            self.bp_low_annots_go_terms,
        ) = self._build_go_terms_tree(
            dataset.ontology.bp_dag.get_valid_terms(),
        )

        (
            self.cc_go_terms_tree,
            self.cc_low_annots_go_terms,
        ) = self._build_go_terms_tree(
            dataset.ontology.cc_dag.get_valid_terms(),
        )

        (
            self.mf_go_terms_tree,
            self.mf_low_annots_go_terms,
        ) = self._build_go_terms_tree(
            dataset.ontology.mf_dag.get_valid_terms(),
        )

        log.info("Building SubmodelDataset tree")
        self.bp_submodels_tree = self._build_submodels_tree(
            self.bp_go_terms_tree,
            BP,
        )
        self.cc_submodels_tree = self._build_submodels_tree(
            self.cc_go_terms_tree,
            CC,
        )
        self.mf_submodels_tree = self._build_submodels_tree(
            self.mf_go_terms_tree,
            MF,
        )

    @typechecked
    def _filter_non_compatible_proteins(self) -> None:
        """_filter_non_compatible_proteins
        Remove proteins with sequences less than 3

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.feature_vector_name == CTRIAD:
            for protein in list(self.dataset.proteins.values()):
                if len(protein.sequence) < 3:
                    log.warn(
                        f"Protein {protein.id} with sequence "
                        f"'{protein.sequence}' has length less than 3. Since "
                        "'ctriad' requires sequence's length greater or equal "
                        "than 3, it will be removed from dataset."
                    )

                    for annot in protein.get_all_annotations():
                        self.dataset.ontology.get_primary_term(
                            annot.go_term_id
                        ).remove_annotation(protein.id)
                    self.dataset.remove_protein(protein.id)

    @typechecked
    def _build_go_terms_tree(
        self, go_terms: list[GOTerm]
    ) -> tuple[dict[int, dict[int, list[GOTerm]]], list[GOTerm]]:
        """_build_go_terms_tree
        Create a dict where GO terms are organized by bucket

        Parameters
        ----------
        go_terms: list
            List of GOTerm's

        Returns
        -------
        Dict of GO terms in useful buckets, list of GO terms in the low annotations bucket
        """
        go_terms_tree = {}
        low_annots_go_terms = []
        for go_term in go_terms:
            bucket = self._get_bucket(go_term)
            if bucket > 0:
                tree_level = go_terms_tree.get(go_term.level, {})
                level_bucket = tree_level.get(bucket, [])
                level_bucket.append(go_term)
                # Handle the case of the entries being 'None' previously
                tree_level[bucket] = level_bucket
                go_terms_tree[go_term.level] = tree_level
            else:
                low_annots_go_terms.append(go_term)
        return go_terms_tree, low_annots_go_terms

    @typechecked
    def _get_bucket(self, go_term: GOTerm) -> int:
        """_get_bucket
        Get the bucket for a given GO term

        Parameters
        ----------
        go_term: GOTerm
            ...

        Returns
        -------
        Integer
        """
        prev_bucket = 0
        for bucket in self.buckets:
            if len(go_term.get_manual_non_obsolete_annotations()) < bucket:
                return prev_bucket
            prev_bucket = bucket
        return prev_bucket

    @typechecked
    def _build_submodels_tree(
        self,
        go_terms_tree: dict[int, dict[int, list[GOTerm]]],
        namespace: str,
    ) -> dict[int, dict[int, list[SubModelDataset]]]:
        """_build_submodels_tree
        This function splits the list of GO Terms into
        'self.go_terms_per_submodel' terms to create the model's output.
        Note that the actual maximum number of GO terms that a network might
        have is '2 * go_terms_per_submodel - 1' when, for example,
        go_terms_per_submodel == 4 and the number of GO terms in a level bucket
        is 7. In this case, 4 terms + 3 reminder terms will be in the same
        network.

        Graphically it looks like:
        {
            1: { # GO level 1
                30: [[x1, x2, x3, x4], [x5, x6, x7, x8] , ...],
                # GO terms in level 1 associated with 30 to 100 protein seqs.
                100: [[y1, y2, y3, ...], [y5, y6, ], ...],
                ...
            }
            2: { # GO level 2
                30: [[w1, w2, ...], [...]],
                100: [z1, z2, ...]
            },
            ...
        }

        Parameters
        ----------
        go_terms_tree: dict
            ...

        Returns
        -------
        submodels_tree: dict

        """
        submodels_tree = {}

        for level, go_terms_level in go_terms_tree.items():
            submodels_tree[level] = {}
            for bucket, go_terms in go_terms_level.items():
                go_terms.sort(
                    key=lambda x: len(x.get_manual_non_obsolete_annotations())
                )
                submodels_tree[level][bucket] = []
                number_of_go_terms = len(go_terms)
                number_of_models = max(
                    1, number_of_go_terms // self.go_terms_per_submodel
                )
                adjusted_number_of_terms_per_model = (
                    number_of_go_terms // number_of_models
                )
                remainder_terms = number_of_go_terms % number_of_models

                offset = 0
                for i in range(0, number_of_models):
                    if i < remainder_terms:
                        number_of_terms_in_model = (
                            adjusted_number_of_terms_per_model + 1
                        )
                    else:
                        number_of_terms_in_model = (
                            adjusted_number_of_terms_per_model
                        )

                    submodel_go_terms = go_terms[
                        offset : offset + number_of_terms_in_model
                    ]
                    submodels_tree[level][bucket].append(
                        self._build_submodel_dataset(
                            submodel_go_terms, namespace, level, bucket, i
                        )
                    )
                    offset += number_of_terms_in_model
                assert offset == number_of_go_terms
        return submodels_tree

    @typechecked
    def _build_submodel_dataset(
        self,
        go_terms: list[GOTerm],
        namespace: str,
        ontology_level: int,
        bucket: int,
        level_index: int,
    ) -> SubModelDataset:
        """_build_submodel_dataset
        Creates the SubModelDataset that will become a DeepredDataset

        - x_{train, val, test}: Feature vector (e.g. ctriad) of sequence.
        For example:
            [
                [0.3, 0.1, 0.74, ..., 0.3],
                ...,
                [0.5, 0.21, 0.94, ..., 0.86],
            ]

        Where the dimensions are (number of samples, length of feature vector)

        - y_{train, val, test}: Labels signaling if sequence is associated
        with submodel GO term. For example:
            [
                [1, 0, 0, 0],
                ...
                [0, 0, 1, 0],
            ]

        Where the dimensions are (number of samples, number of output GO terms)

        Parameters
        ----------
        go_terms: list
            List of GOTerms
        ontology_level: int
            Level
        bucket: int
            Bucket number
        level_index: int

        Returns
        -------
        SubModelDataset
        """
        # Before building 'y', we build 'y_dict' since we need to keep
        # track of the PIDs so that we can merge annotations that have same PID
        # but different GO Terms. The 'label' list in 'y' and 'y_dict' is a list of
        # 0's and 1's that indicates if a given protein has a given GO annotation
        y_dict = {}
        for i, go_term in enumerate(go_terms):
            annotations = go_term.get_manual_non_obsolete_annotations()
            for annot in annotations:
                pid = annot.protein_id
                y = y_dict.get(pid)
                if y:
                    y_dict[pid][i] = 1
                else:
                    label = [0] * len(go_terms)
                    label[i] = 1
                    y_dict[pid] = label

        samples_pid = np.zeros(len(y_dict), dtype="str")
        X = np.zeros(
            (len(y_dict), FEATURE_VECTOR_LENGTHS[self.feature_vector_name])
        )
        y = np.zeros((len(y_dict), len(go_terms)))
        for row, (pid, label) in enumerate(y_dict.items()):
            y[row] = label
            X[row] = self.feature_vectors[pid]
            samples_pid[row] = pid

        # Balance dataset using MLSMOTE
        mlsmote = MLSMOTE([False] * y.shape[1])
        mean_ir = utils.calculate_mean_imbalance_ratio(y)
        X_resampled = np.array(X)
        y_resampled = np.array(y)
        if mean_ir > self.target_imbalance_ratio:
            X_resampled, y_resampled = mlsmote.fit_resample(X, y)
            resampled_mean_ir = utils.calculate_mean_imbalance_ratio(
                y_resampled
            )
            log.info(
                f"Model: {self.submodel_global_index}, "
                f"{namespace}-{ontology_level}-{bucket}-{level_index} being "
                f"resampled. Mean IR: {mean_ir:.2f}. "
                f"New mean IR: {resampled_mean_ir:.2f}."
            )
        # We need to make sure that there at least one label of each class in
        # all train, val and test.
        labels_present_in_train_val_test = False
        tries = 3
        while not labels_present_in_train_val_test:
            # Shuffle rows of 'X' and 'y' in unison
            rng_state = np.random.get_state()
            np.random.shuffle(X_resampled)
            np.random.set_state(rng_state)
            np.random.shuffle(y_resampled)

            train_samples = round(len(y_dict) * self.training_partition)
            validation_samples = round(len(y_dict) * self.validation_partition)

            X_train = X_resampled[:train_samples, :]
            X_val = X_resampled[
                train_samples : train_samples + validation_samples, :
            ]
            X_test = X_resampled[train_samples + validation_samples :, :]
            y_train = y_resampled[:train_samples, :]
            y_val = y_resampled[
                train_samples : train_samples + validation_samples, :
            ]
            y_test = y_resampled[train_samples + validation_samples :, :]

            train_non_zero = np.all(np.count_nonzero(y_train, axis=0))
            val_non_zero = np.all(np.count_nonzero(y_val, axis=0))
            test_non_zero = np.all(np.count_nonzero(y_test, axis=0))

            if train_non_zero and val_non_zero and test_non_zero:
                labels_present_in_train_val_test = True
            else:
                tries -= 1
                if tries == 0:
                    log.warn(
                        "After 3 tries, the dataset couldn't be split "
                        "such that labels from each class were present on "
                        "all training, validation and test datasets."
                        " Saving labels 'y.csv' in current directory"
                    )
                    np.savetxt("y.csv", y_resampled, fmt="%d", delimiter=",")
                    exit(1)

        submodel_dataset = SubModelDataset(
            go_terms,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X,
            y,
            samples_pid,
            self.feature_vector_name,
            self.submodel_global_index,
            namespace,
            ontology_level,
            bucket,
            level_index,
        )
        self.submodel_global_index += 1
        return submodel_dataset

    def _build_feature_vectors(self) -> dict[str, list[float]]:
        sequences = []
        proteins_ids = []

        for protein in self.dataset.proteins.values():
            sequences.append(protein.sequence)
            proteins_ids.append(protein.id)

        feature_vectors = utils.get_protein_feature(
            self.feature_vector_name, sequences, proteins_ids
        )
        feature_dict = {}
        for pid, feature_vector in zip(proteins_ids, feature_vectors):
            feature_dict[pid] = feature_vector
        return feature_dict

    @typechecked
    def write_datasets_to_files(self, output_dir: Path):
        self._write_submodel_dataset_tree_to_files(
            self.bp_submodels_tree, output_dir
        )
        self._write_submodel_dataset_tree_to_files(
            self.cc_submodels_tree, output_dir
        )
        self._write_submodel_dataset_tree_to_files(
            self.mf_submodels_tree, output_dir
        )

    @typechecked
    def _write_submodel_dataset_tree_to_files(
        self,
        submodels_dataset: dict[int, dict[int, list[SubModelDataset]]],
        output_dir: Path,
    ):
        """_write_submodel_dataset_tree_to_files
        Write submodel datasets to files

        Parameters
        ----------
        namespace: str
            GO namespace
        submodel_dataset: dict of lists
            Multiple datasets
        output_dir: Path
            Output file location

        Returns
        -------
        None
        """
        os.makedirs(output_dir, exist_ok=True)
        for buckets in submodels_dataset.values():
            for submodels in buckets.values():
                for submodel in submodels:
                    submodel.write_dataset_to_files(output_dir)

    @typechecked
    def write_datasets_to_hdf5(self, output_dir: Path):
        """write_datasets_to_hdf5
        Prepare to write submodel datasets to HD5 for all namespaces

        Parameters
        ----------
        output_dir: Path
            Output file location

        Returns
        -------
        None
        """
        os.makedirs(output_dir, exist_ok=True)
        hf = h5py.File(output_dir / "protcast_dataset.hdf5", "w", driver="core")
        self.write_submodel_dataset_tree_to_hdf5(hf, self.bp_submodels_tree)
        self.write_submodel_dataset_tree_to_hdf5(hf, self.cc_submodels_tree)
        self.write_submodel_dataset_tree_to_hdf5(hf, self.mf_submodels_tree)

    def write_submodel_dataset_tree_to_hdf5(
        self, hdf5_file, submodels_dataset
    ):
        """write_submodel_dataset_tree_to_hdf5
        Prepare to write submodel datasets to HD5 for one namespace

        Parameters
        ----------
        hdf5file: str
            HDF5 content
        namespace: str
            GO namespace
        submodels_dataset: dict of datasets

        Returns
        -------
        None
        """
        for buckets in submodels_dataset.values():
            for submodels in buckets.values():
                for submodel in submodels:
                    submodel.write_dataset_to_hdf5(hdf5_file)

    @classmethod
    @typechecked
    def from_serialized_protcast_dataset(
        cls, model_dataset_path: Path
    ) -> DeepredDataset:
        """from_serialized_protcast_dataset
        Deserialize a DeepredDataset

        Parameters
        ----------
        model_dataset_path: Path
            Pickle location

        Returns
        -------
        DeepredDataset
        """
        with open(model_dataset_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_files(
        cls,
        ontology_path,
        swissprot_t0_path,
        swissprot_t1_path,
        trembl_path,
        goa_path,
    ) -> DeepredDataset:
        """from_files
        Create DeepredDataset from Dataset given GO and sequence file paths

        Parameters
        ----------
        ontology_path: Path
            Path to GO ontology file
        swissprot_t0_path:
            Path to Swissprot t0
        swissprot_t1_path:
            Path to Swisssprot t1
        trembl_path: Path
            Path to trembl file
        goa_path: Path
            Path to GO GOA file

        Returns
        -------
        DeepredDataset
        """
        dataset = Dataset(
            ontology_path,
            swissprot_t0_path,
            swissprot_t1_path,
            trembl_path,
            goa_path,
        )
        return DeepredDataset(dataset)

    @typechecked
    def save(self, output_dir: Path) -> None:
        """save
        Serialize a DeepredDataset as a *bin file

        Parameters
        ----------
        output_dir: Path
           Path to *protcast_dataset.bin file

        Returns
        -------
        None
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / Path("protcast_dataset.bin"), "wb") as f:
            pickle.dump(self, f)

    def summary(self):
        """summary
        Print a summary of a DeepredDataset

        The summary contains:
        - The GO Terms tree lengths.
        - The GO Terms tree terms.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pp = pprint.PrettyPrinter(indent=4)
        p_bp_go_terms_tree, p_bp_go_terms_tree_lengths = summarize_tree(
            self.bp_go_terms_tree
        )
        p_cc_go_terms_tree, p_cc_go_terms_tree_lengths = summarize_tree(
            self.cc_go_terms_tree
        )
        p_mf_go_terms_tree, p_mf_go_terms_tree_lengths = summarize_tree(
            self.mf_go_terms_tree
        )

        print("---------- Deepred Dataset -----------")
        print("------- BP GO Terms Tree Lengths -------")
        pp.pprint(p_bp_go_terms_tree_lengths)
        print("------- BP Submodels -------")
        print_submodels_tree(BP, self.bp_submodels_tree)
        print("------- CC GO Terms Tree Lengths -------")
        pp.pprint(p_cc_go_terms_tree_lengths)
        print("------- CC Submodels -------")
        print_submodels_tree(CC, self.bp_submodels_tree)
        print("------- MF GO Terms Tree Lengths -------")
        pp.pprint(p_mf_go_terms_tree_lengths)
        print("------- MF Submodels -------")
        print_submodels_tree(MF, self.mf_submodels_tree)
        print("-------- BP GO Terms Tree Terms --------")
        pp.pprint(p_bp_go_terms_tree)
        print("-------- CC GO Terms Tree Terms --------")
        pp.pprint(p_cc_go_terms_tree)
        print("-------- MF GO Terms Tree Terms --------")
        pp.pprint(p_mf_go_terms_tree)
        print("--------- End Deepred Dataset --------")


def summarize_tree(go_terms_tree):
    """summarize_tree
    Iterate over a DeepredDataset prior to summarizing it

    Parameters
    ----------
    go_terms_tree: dict
        ...

    Returns
    -------
    Dicts of GO terms and GO term tree lengths
    """
    p_go_terms_tree = {}
    p_go_terms_tree_lengths = {}
    for level, buckets in go_terms_tree.items():
        p_go_terms_tree[level] = {}
        p_go_terms_tree_lengths[level] = {}
        for bucket, go_terms in buckets.items():
            p_go_terms_tree[level][bucket] = []
            p_go_terms_tree_lengths[level][bucket] = len(go_terms)
            for go_term in go_terms:
                p_go_terms_tree[level][bucket].append(go_term.id)
    return p_go_terms_tree, p_go_terms_tree_lengths


def print_submodels_tree(namespace, submodels_tree):
    """print_submodels_tree
    Prints out namespace, level, bucket, and GO terms

    Parameters
    ----------
    namespace: str
        GO namespace
    submodels_tree: dict
        ...

    Returns
    -------
    None
    """
    print("Namespace\tLevel\tBucket\tGo Terms")
    for level, buckets in submodels_tree.items():
        for bucket, submodels in buckets.items():
            for model in submodels:
                print_submodel(namespace, level, bucket, model)


def print_submodel(namespace, level, bucket, go_terms):
    """print_submodel
    Prints out GO terms by namespace, level, bucket

    Parameters
    ----------
    namespace: str
        GO namespace
    level: int
        GO level
    bucket: int
        Bucket in model
    go_terms: list
        List of GO terms

    Returns
    -------
    None
    """
    ids = [x.id for x in go_terms]
    print(f"{namespace}\t{level}\t{bucket}\t{','.join(ids)}")

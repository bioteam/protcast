from protcast import BP, CC, MF
from protcast.preprocessing import utils
from protcast.preprocessing.simple_dataset import SimpleDataset
from protcast.preprocessing.deepred_dataset import (
    DeepredDataset,
    SubModelDataset,
)
from protcast.preprocessing.ontology import GODAG
import logging as log
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
from typeguard import typechecked

MEAN_IMBALANCE_RATIO_THRESHOLD = 3


@typechecked
def generate_dataset_stats(
    dataset: SimpleDataset,
    output_dir: Path,
):
    """generate_dataset_stats
    Create dataset_general.txt, cc_go_terms.txt, bp_go_terms.txt,
    mf_go_terms.txt, obsolete_bp_go_terms.txt, obsolete_cc_go_terms.txt,
    and obsolete_mf_go_terms.txt files.

    Parameters
    ----------
    dataset: Dataset object
        ...
    output_dir: Path
        Path to output directory

    Returns
    -------
    None
    """
    with open(output_dir / Path("dataset_general.txt"), "w") as f:
        f.write(f"Creation Time: {dataset.created_at}\n")
        f.write(
            f"Ontology File: {dataset.ontology_path} (md5: "
            f"{dataset.ontology_md5})\n"
        )
        f.write(
            f"Swissprot t0 File: {dataset.swissprot_t0_path} (md5: "
            f"{dataset.swissprot_t0_md5})\n"
        )
        f.write(
            f"Swissprot t1 File: {dataset.swissprot_t1_path} (md5: "
            f"{dataset.swissprot_t1_md5})\n"
        )
        f.write(f"GOA File: {dataset.goa_path} (md5: {dataset.goa_md5})\n")

    with open(output_dir / Path("bp_go_terms.txt"), "w") as f:
        f.write(dataset.ontology.bp_dag.to_text())

    with open(output_dir / Path("cc_go_terms.txt"), "w") as f:
        f.write(dataset.ontology.cc_dag.to_text())

    with open(output_dir / Path("mf_go_terms.txt"), "w") as f:
        f.write(dataset.ontology.mf_dag.to_text())

    def write_obsolete_go_terms(file_name: Path, dag: GODAG):
        with open(output_dir / file_name, "w") as f:
            f.write("Term\n")
            f.write(
                "\n".join([f"{node.id}" for node in dag.get_obsolete_nodes()])
            )
            f.write("\n")

    write_obsolete_go_terms(
        Path("obsolete_bp_go_terms.txt"), dataset.ontology.bp_dag
    )
    write_obsolete_go_terms(
        Path("obsolete_cc_go_terms.txt"), dataset.ontology.cc_dag
    )
    write_obsolete_go_terms(
        Path("obsolete_mf_go_terms.txt"), dataset.ontology.mf_dag
    )

    with open(output_dir / Path("go_terms_not_found.txt"), "w") as f:
        f.write(
            "This is a list of GO terms that were not found in the ontology "
            "when parsing Swissprot. Count: "
            f"{len(dataset.go_terms_not_found)}\n"
        )
        f.write("\n".join(dataset.go_terms_not_found))


@typechecked
def generate_protcast_dataset_stats(
    protcast_dataset: DeepredDataset, output_dir: Path
):
    """generate_protcast_dataset_stats
    Create protcast_general.txt file.

    Parameters
    ----------
    protcast_dataset: DeepredDataset object
        ...
    output_dir: Path
        ...

    Returns
    -------
    None
    """
    generate_dataset_stats(protcast_dataset.dataset, output_dir)

    (
        bp_number_of_samples_submodels_ds,
        bp_imbalanced_models,
    ) = _generate_namespace_submodel_dataset_stats(
        BP, protcast_dataset.bp_submodels_tree, output_dir
    )
    (
        cc_number_of_samples_submodels_ds,
        cc_imbalanced_models,
    ) = _generate_namespace_submodel_dataset_stats(
        CC, protcast_dataset.cc_submodels_tree, output_dir
    )
    (
        mf_number_of_samples_submodels_ds,
        mf_imbalanced_models,
    ) = _generate_namespace_submodel_dataset_stats(
        MF, protcast_dataset.mf_submodels_tree, output_dir
    )

    _generate_namespace_submodel_classes_histogram(
        BP, bp_number_of_samples_submodels_ds, output_dir
    )
    _generate_namespace_submodel_classes_histogram(
        CC, cc_number_of_samples_submodels_ds, output_dir
    )
    _generate_namespace_submodel_classes_histogram(
        MF, mf_number_of_samples_submodels_ds, output_dir
    )

    with open(output_dir / Path("protcast_general.txt"), "w") as f:
        f.write(
            f"DeepredDataset Creation Time: {protcast_dataset.created_at}\n"
        )

        total_number_of_submodels = protcast_dataset.submodel_global_index
        f.write("Total number of models: " f"{total_number_of_submodels}\n")

        imbalanced_models = (
            len(bp_imbalanced_models)
            + len(cc_imbalanced_models)
            + len(mf_imbalanced_models)
        )
        f.write(
            f"Number of imbalanced models: {imbalanced_models} "
            f"({imbalanced_models/total_number_of_submodels*100}%)\n"
        )

        def list_imbalanaced_submodels(
            namespace: str, imbalanced_models: list
        ):
            f.write(f"Imbalanced Models ({namespace}):\n")
            f.write("\n".join(imbalanced_models))

        list_imbalanaced_submodels(BP, bp_imbalanced_models)
        list_imbalanaced_submodels(CC, cc_imbalanced_models)
        list_imbalanaced_submodels(MF, mf_imbalanced_models)


@typechecked
def _generate_namespace_submodel_dataset_stats(
    namespace_name: str,
    namespace_submodels_dataset: dict[int, dict[int, list[SubModelDataset]]],
    output_dir: Path,
) -> tuple[list[tuple[str, int]], set[str]]:
    """_generate_namespace_submodel_dataset_stats
    Calculate basis stats on submodels including IR measures and
    return the statistics and the statistics on imbalanced models.

    Parameters
    ----------
    namespace_name: str
        ...
    namespace_submodels_dataset: dict of dicts
        Values are SubModelDataset objects

    Returns
    -------
    number_of_samples_submodels_ds: list of strings
    imbalanced_models: set of strings
    """
    with open(
        output_dir / Path(f"protcast_{namespace_name}_submodels.txt"),
        "w",
    ) as f:
        number_of_samples_submodels_ds = []
        imbalanced_models = set()
        f.write(
            f"Mean Imbalanced Ratio Threshold: "
            f"{MEAN_IMBALANCE_RATIO_THRESHOLD}\n"
        )
        for level, buckets in sorted(namespace_submodels_dataset.items()):
            for bucket, datasets in sorted(buckets.items()):
                for dataset in datasets:
                    samples = dataset.X_train.shape[0]
                    level_index = dataset.level_index
                    mean_ir = utils.calculate_mean_imbalance_ratio(
                        np.vstack(
                            (
                                dataset.y_train,
                                dataset.y_val,
                                dataset.y_test,
                            )
                        )
                    )

                    if mean_ir == float("inf"):
                        log.warn(
                            f"Submodel: {level}-{bucket}-{level_index} has an "
                            "undefined mean IR. Writing dataset to file"
                        )
                        dataset.write_dataset_to_files(
                            namespace_name, bucket, output_dir
                        )

                    f.write(
                        f"Submodel {dataset.global_index}, level: "
                        f"{dataset.ontology_level}, "
                        f"bucket: {bucket}, subindex: {level_index}\n"
                        f"Number of GO Terms: {len(dataset.go_terms)}\n"
                        f"GO Terms: {[got.id for got in dataset.go_terms]}\n"
                        f"Number of samples: {samples}\n"
                        f"Mean IR: {mean_ir:.3f}\n"
                        "\n"
                    )
                    number_of_samples_submodels_ds.append(
                        (f"{level}-{bucket}-{level_index}", samples)
                    )
                    if mean_ir > MEAN_IMBALANCE_RATIO_THRESHOLD:
                        imbalanced_models.add(
                            f"{level} - {bucket} - {level_index}"
                        )
        return number_of_samples_submodels_ds, imbalanced_models


@typechecked
def _generate_namespace_submodel_classes_histogram(
    namespace: str,
    number_of_samples_submodel_ds: list[tuple[str, int]],
    output_dir: Path,
) -> None:
    """_generate_namespace_submodel_classes_histogram
    ...

    Parameters
    ----------
    namespace: str
        GO namespace
    number_of_samples_submodel_ds: list of strings
        Basic statistics on a submodel
    output_dir: Path
        ...

    Returns
    -------
    None
    """
    fig = plt.figure()
    plt.bar(
        [x[0] for x in number_of_samples_submodel_ds],
        [x[1] for x in number_of_samples_submodel_ds],
    )
    plt.title(f"Number of samples in {namespace} SubModels")
    plt.xlabel("Model identifier")
    plt.ylabel("Number of samples")
    pickle.dump(
        fig,
        open(
            output_dir
            / Path(f"number_of_samples_per_submodel_{namespace}.fig.pickle"),
            "wb",
        ),
    )
    plt.savefig(
        output_dir / Path(f"number_of_samples_per_submodel_{namespace}.png")
    )

# from protcast import BP, CC, MF
# from protcast.preprocessing import utils
from protcast.preprocessing.simple_dataset import SimpleDataset
from protcast.preprocessing.ontology import GODAG
import matplotlib.pyplot as plt
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
    dataset: SimpleDataset
        ...
    output_dir: Path
        Path to output directory

    Returns
    -------
    None
    """
    with open(output_dir / Path("simpledataset_general.txt"), "w") as f:
        f.write(f"Creation Time: {dataset.created_at}\n")
        f.write(
            f"Ontology File: {dataset.ontology_path} (md5: "
            f"{dataset.ontology_md5})\n"
        )
        f.write(
            f"Swissprot file: {dataset.swissprot_path} (md5: "
            f"{dataset.swissprot_md5})\n"
        )
        f.write(f"GOA File: {dataset.gaf_path}\n")
        bp_levels = list()
        cc_levels = list()
        mf_levels = list()
        for level in range(11):
            bp_levels.append(str(len(dataset.ontology.bp_dag.get_nodes_by_level(level))))
        for level in range(11):
            cc_levels.append(str(len(dataset.ontology.cc_dag.get_nodes_by_level(level))))
        for level in range(11):
            mf_levels.append(str(len(dataset.ontology.mf_dag.get_nodes_by_level(level))))
        f.write("\nNodes by level (0-10)\n")
        f.write("BP\t" + "\t".join(bp_levels) + "\n")
        f.write("CC\t" + "\t".join(cc_levels) + "\n")
        f.write("MF\t" + "\t".join(mf_levels) + "\n")

    with open(output_dir / Path("bp_go_terms.txt"), "w") as f:
        f.write(dataset.ontology.bp_dag.to_text())

    with open(output_dir / Path("cc_go_terms.txt"), "w") as f:
        f.write(dataset.ontology.cc_dag.to_text())

    with open(output_dir / Path("mf_go_terms.txt"), "w") as f:
        f.write(dataset.ontology.mf_dag.to_text())

    def write_obsolete_go_terms(file_name: Path, dag: GODAG):
        with open(output_dir / file_name, "w") as f:
            f.write(
                "\n".join([f"{node.id}" for node in dag.get_obsolete_nodes()])
            )

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
        f.write(f"Count: {len(dataset.go_terms_not_found)}\n")
        f.write("\n".join(dataset.go_terms_not_found))

    # _generate_namespace_submodel_classes_histogram(
    #     BP, bp_number_of_samples_submodels_ds, output_dir
    # )
 



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

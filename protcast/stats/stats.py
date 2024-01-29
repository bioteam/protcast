from protcast import BP, CC, MF
from protcast.preprocessing.simple_dataset import SimpleDataset
from protcast.preprocessing.ontology import GODAG
import matplotlib.pyplot as plt
from pathlib import Path
from typeguard import typechecked


@typechecked
def generate_dataset_stats(
    dataset_location: str,
):
    """generate_dataset_stats
    Create dataset_general.txt, cc_go_terms.txt, bp_go_terms.txt,
    mf_go_terms.txt, obsolete_bp_go_terms.txt, obsolete_cc_go_terms.txt,
    and obsolete_mf_go_terms.txt files.

    Parameters
    ----------
    dataset: str
        Location of serialized SimpleDataset

    Returns
    -------
    None
    """
    dataset = SimpleDataset.from_serialized_file(dataset_location)
    output_dir = Path(dataset_location).parent

    bp_levels = list()
    cc_levels = list()
    mf_levels = list()
    
    with open(output_dir / Path("simpledataset_statistics.txt"), "w") as f:
        f.write(f"Creation Time: {dataset.created_at}\n")
        f.write(
            f"Ontology File: {dataset.ontology_path} (md5: "
            f"{dataset.ontology_md5})\n"
        )
        f.write(
            f"Swissprot file: {dataset.swissprot_path} (md5: "
            f"{dataset.swissprot_md5})\n"
        )
        f.write(f"GOA File: {dataset.gaf_path}\n\n")

        f.write(f"Nodes in BP: {len(dataset.ontology.bp_dag.nodes.keys())}\n")
        f.write(f"Nodes in CC: {len(dataset.ontology.cc_dag.nodes.keys())}\n")
        f.write(f"Nodes in MF: {len(dataset.ontology.mf_dag.nodes.keys())}\n\n")

        f.write(f"Annotations in BP: {len(dataset.ontology.bp_dag.annotations)}\n")
        f.write(f"Annotations in CC: {len(dataset.ontology.cc_dag.annotations)}\n")
        f.write(f"Annotations in MF: {len(dataset.ontology.mf_dag.annotations)}\n\n")

        for level in range(12):
            bp_levels.append(len(dataset.ontology.bp_dag.get_nodes_by_level(level)))
        for level in range(12):
            cc_levels.append(len(dataset.ontology.cc_dag.get_nodes_by_level(level)))
        for level in range(12):
            mf_levels.append(len(dataset.ontology.mf_dag.get_nodes_by_level(level)))

        f.write("Nodes by level (0-11)\n")
        f.write("BP\t" + "\t".join([str(x) for x in bp_levels]) + "\n")
        f.write("CC\t" + "\t".join([str(x) for x in cc_levels]) + "\n")
        f.write("MF\t" + "\t".join([str(x) for x in mf_levels]) + "\n")

    with open(output_dir / Path("bp_go_terms.tsv"), "w") as f:
        f.write("Term\tLevel\tAnnotations\tManual Annotations,Is_obsolete\tPrimary id") 
        f.write(dataset.ontology.bp_dag.to_text())

    with open(output_dir / Path("cc_go_terms.tsv"), "w") as f:
        f.write("Term\tLevel\tAnnotations\tManual Annotations,Is_obsolete\tPrimary id") 
        f.write(dataset.ontology.cc_dag.to_text())

    with open(output_dir / Path("mf_go_terms.tsv"), "w") as f:
        f.write("Term\tLevel\tAnnotations\tManual Annotations,Is_obsolete\tPrimary id") 
        f.write(dataset.ontology.mf_dag.to_text())

    generate_ontology_levels_histogram(
        BP, bp_levels, output_dir
    )
    generate_ontology_levels_histogram(
        CC, cc_levels, output_dir
    )
    generate_ontology_levels_histogram(
        MF, mf_levels, output_dir
    )

    def write_obsolete_go_terms(file_name: Path, dag: GODAG):
        with open(file_name, "w") as f:
            f.write(
                "\n".join([f"{node.id}" for node in dag.get_obsolete_nodes()])
            )

    write_obsolete_go_terms(
        Path(output_dir/"obsolete_bp_go_terms.txt"), dataset.ontology.bp_dag
    )
    write_obsolete_go_terms(
        Path(output_dir/"obsolete_cc_go_terms.txt"), dataset.ontology.cc_dag
    )
    write_obsolete_go_terms(
        Path(output_dir/"obsolete_mf_go_terms.txt"), dataset.ontology.mf_dag
    )

    with open(output_dir / Path("go_terms_not_found.txt"), "w") as f:
        f.write(f"Count: {len(dataset.go_terms_not_found)}\n")
        f.write("\n".join(dataset.go_terms_not_found))


@typechecked
def generate_ontology_levels_histogram(
    namespace: str,
    levels: list[int],
    output_dir: Path,
) -> None:
    """generate_ontology_levels_histogram
    ...

    Parameters
    ----------
    namespace: str
        GO namespace
    levels: list of ints
        Number at each level
    output_dir: Path
        ...

    Returns
    -------
    None
    """
    plt.bar(range(12), levels)
    plt.title(f"Number of terms per level in {namespace}")
    plt.xlabel("Level")
    plt.ylabel("Number of terms")
    plt.savefig(
        output_dir / Path(f"number_terms_per_level_in_{namespace}.png")
    )

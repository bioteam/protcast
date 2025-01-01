from protcast.preprocessing.protcast_dataset import ProtCastDataset
from protcast.globals import CC, BP, MF
from pathlib import Path
from typeguard import typechecked
import pandas as pd
import plotly.express as px


@typechecked
def create_stats_files(dataset_location: str) -> None:
    """create_stats_files
    Create ProtCastDataset_statistics.txt, cc_go_terms.txt, bp_go_terms.txt,
    mf_go_terms.txt files and histograms of terms, annotations, and levels.

    Parameters
    ----------
    dataset: str
        Location of serialized ProtCastDataset

    Returns
    -------
    None
    """
    dataset = ProtCastDataset.load_serialized_file(dataset_location)
    output_dir = Path(dataset_location).parent
    # No terms at level 14
    num_levels = 14

    bp_node_levels = [0] * num_levels
    cc_node_levels = [0] * num_levels
    mf_node_levels = [0] * num_levels
    bp_annot_levels = [0] * num_levels
    cc_annot_levels = [0] * num_levels
    mf_annot_levels = [0] * num_levels

    with open(output_dir / Path("ProtCastDataset_statistics.txt"), "w") as f:
        """Make ProtCastDataset.statistics.txt file"""
        f.write(f"Creation Time: {dataset.created_at}\n")
        f.write(
            f"Ontology file: {dataset.ontology_path} (md5: "
            f"{dataset.ontology_md5})\n"
        )
        f.write(
            f"Swissprot file: {dataset.swissprot_path} (md5: "
            f"{dataset.swissprot_md5})\n"
        )
        f.write(f"GOA file: {dataset.gaf_path}\n")
        f.write(f"Trembl file: {dataset.trembl_path}\n\n")

        bp_nodes = dataset.get_all_terms(namespace=BP)
        cc_nodes = dataset.get_all_terms(namespace=CC)
        mf_nodes = dataset.get_all_terms(namespace=MF)

        f.write(f"Nodes in {BP}: {len(bp_nodes)}\n")
        f.write(f"Nodes in {CC}: {len(cc_nodes)}\n")
        f.write(f"Nodes in {MF}: {len(mf_nodes)}\n\n")

        f.write(
            f"Annotations in {BP}: {len(dataset.get_all_annotations(namespace=BP))}\n"
        )
        f.write(
            f"Annotations in {CC}: {len(dataset.get_all_annotations(namespace=CC))}\n"
        )
        f.write(
            f"Annotations in {MF}: {len(dataset.get_all_annotations(namespace=MF))}\n\n"
        )

        for t in bp_nodes:
            bp_annot_levels[t.level] += len(t.annotations)
            bp_node_levels[t.level] += 1
        for t in cc_nodes:
            cc_annot_levels[t.level] += len(t.annotations)
            cc_node_levels[t.level] += 1
        for t in mf_nodes:
            mf_annot_levels[t.level] += len(t.annotations)
            mf_node_levels[t.level] += 1

        f.write("Nodes by level (0-13)\n")
        f.write(BP + "\t" + "\t".join([str(x) for x in bp_node_levels]) + "\n")
        f.write(CC + "\t" + "\t".join([str(x) for x in cc_node_levels]) + "\n")
        f.write(
            MF + "\t" + "\t".join([str(x) for x in mf_node_levels]) + "\n\n"
        )

        f.write("Annotations by level (0-13)\n")
        f.write(
            BP + "\t" + "\t".join([str(x) for x in bp_annot_levels]) + "\n"
        )
        f.write(
            CC + "\t" + "\t".join([str(x) for x in cc_annot_levels]) + "\n"
        )
        f.write(
            MF + "\t" + "\t".join([str(x) for x in mf_annot_levels]) + "\n"
        )

        """Make 'GO Terms by Level' figure"""
        df = pd.DataFrame(
            {"BP": bp_node_levels, "CC": cc_node_levels, "MF": mf_node_levels}
        )
        fig = px.bar(
            df,
            x=df.index,
            y=["BP", "CC", "MF"],
            barmode="stack",
            title="GO Terms by Level",
            text_auto=True,
        )
        fig.update_layout(xaxis_title="Level", yaxis_title="Number of Terms")
        fig.show()
        fig.write_image(output_dir / "GO_terms_by_level.png")

        """Make 'Annotations by Level' figure"""
        df = pd.DataFrame(
            {
                "BP": bp_annot_levels,
                "CC": cc_annot_levels,
                "MF": mf_annot_levels,
            }
        )
        fig = px.bar(
            df,
            x=df.index,
            y=["BP", "CC", "MF"],
            barmode="stack",
            title="Annotations by Level",
            text_auto=True,
        )
        fig.update_layout(
            xaxis_title="Level", yaxis_title="Number of Annotations"
        )
        fig.show()
        fig.write_image(output_dir / "annotations_by_level.png")

    """GO term data"""
    namespace_file_map = {
        BP: open(output_dir / Path("bp_go_terms.tsv"), "w"),
        CC: open(output_dir / Path("cc_go_terms.tsv"), "w"),
        MF: open(output_dir / Path("mf_go_terms.tsv"), "w"),
    }

    for fh in namespace_file_map.values():
        fh.write(
            "go_id\talt_id\tname\tlevel\tdepth\t# annotations\t# manual annotations\t# nodes in subgraph\t#seqs in subgraph\n"
        )

    for key, go_term in dataset.annotated_dag.go_terms_map.items():
        num_nodes_subgraph, num_seqs_subgraph = get_subgraph_data(
            go_term, dataset
        )
        namespace_file_map[go_term.namespace].write(
            go_term.go_id
            + "\t"
            + key
            + "\t"
            + go_term.name
            + "\t"
            + str(go_term.level)
            + "\t"
            + str(go_term.depth)
            + "\t"
            + str(len(go_term.annotations))
            + "\t"
            + str(len([x for x in go_term.annotations if x.is_manual is True]))
            + "\t"
            + str(num_nodes_subgraph)
            + "\t"
            + str(num_seqs_subgraph)
            + "\n"
        )

    """GO Terms not found"""
    with open(output_dir / Path("go_terms_not_found.txt"), "w") as f:
        f.write("\n".join(dataset.go_terms_not_found))

    """Annotation data"""
    namespace_file_map = {
        BP: open(output_dir / Path("bp_annotations.tsv"), "w"),
        CC: open(output_dir / Path("cc_annotations.tsv"), "w"),
        MF: open(output_dir / Path("mf_annotations.tsv"), "w"),
    }

    for fh in namespace_file_map.values():
        fh.write("protein_id\tgo_id\tevidence_code\tis_manual\n")

    for go_term in dataset.annotated_dag.go_terms_map.values():
        if go_term.get_all_annotations():
            for annot in go_term.get_all_annotations():
                namespace_file_map[go_term.namespace].write(
                    annot.protein_id
                    + "\t"
                    + annot.go_id
                    + "\t"
                    + annot.evidence_code
                    + "\t"
                    + str(annot.is_manual)
                    + "\n"
                )


def get_subgraph_data(go_term, dataset):
    subgraph_terms = dataset.get_terms(dataset.get_subgraph(go_term.go_id))
    all_annots = list()
    for go_term in subgraph_terms:
        annots = go_term.get_all_annotations()
        if annots:
            all_annots.extend(annots)
    subgraph_seq_ids = [annot.protein_id for annot in all_annots]
    return len(subgraph_terms), len(subgraph_seq_ids)

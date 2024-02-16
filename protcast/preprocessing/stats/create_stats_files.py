from protcast.preprocessing.simple_dataset import SimpleDataset
from protcast.globals import CC, BP, MF
from pathlib import Path
from typeguard import typechecked
import pandas as pd
import plotly.express as px


@typechecked
def create_stats_files(dataset_location: str):
    """create_stats_files
    Create SimpleDataset_statistics.txt, cc_go_terms.txt, bp_go_terms.txt,
    mf_go_terms.txt files and histograms of terms, annotations, and levels.

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

    bp_node_levels = [0] * 14
    cc_node_levels = [0] * 14
    mf_node_levels = [0] * 14
    bp_annot_levels = [0] * 14
    cc_annot_levels = [0] * 14
    mf_annot_levels = [0] * 14
    # bp_zero_names = list()
    # cc_zero_names = list()
    # mf_zero_names = list()

    with open(output_dir / Path("SimpleDataset_statistics.txt"), "w") as f:
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

        # Nodes and Annotations by level
        for t in bp_nodes:
            bp_annot_levels[t.level] += len(t.annotations)
            bp_node_levels[t.level] += 1
            # if t.level == 0:
            #     bp_zero_names.append(t.name)
        for t in cc_nodes:
            cc_annot_levels[t.level] += len(t.annotations)
            cc_node_levels[t.level] += 1
            # if t.level == 0:
            #     cc_zero_names.append(t.name)
        for t in mf_nodes:
            mf_annot_levels[t.level] += len(t.annotations)
            mf_node_levels[t.level] += 1
            # if t.level == 0:
            #     mf_zero_names.append(t.name)

        # f.write(f"{BP} level 0: {','.join(bp_zero_names)}\n")
        # f.write(f"{CC} level 0: {','.join(cc_zero_names)}\n")
        # f.write(f"{MF} level 0: {','.join(mf_zero_names)}\n\n")

        f.write("Nodes by level (0-13)\n")
        f.write(BP + "\t" + "\t".join([str(x) for x in bp_node_levels]) + "\n")
        f.write(CC + "\t" + "\t".join([str(x) for x in cc_node_levels]) + "\n")
        f.write(MF + "\t" + "\t".join([str(x) for x in mf_node_levels]) + "\n\n")

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

        f.write("Annotations by level (0-13)\n")
        f.write(BP + "\t" + "\t".join([str(x) for x in bp_annot_levels]) + "\n")
        f.write(CC + "\t" + "\t".join([str(x) for x in cc_annot_levels]) + "\n")
        f.write(MF + "\t" + "\t".join([str(x) for x in mf_annot_levels]) + "\n")

        df = pd.DataFrame(
            {"BP": bp_annot_levels, "CC": cc_annot_levels, "MF": mf_annot_levels}
        )
        fig = px.bar(
            df,
            x=df.index,
            y=["BP", "CC", "MF"],
            barmode="stack",
            title="Annotations by Level",
            text_auto=True,
        )
        fig.update_layout(xaxis_title="Level", yaxis_title="Number of Annotations")
        fig.show()
        fig.write_image(output_dir / "annotations_by_level.png")

    with open(output_dir / Path("bp_go_terms.tsv"), "w") as f:
        f.write("Term\tName\tLevel\tDepth\tAnnotations\tManual Annotations\n")
        for node in bp_nodes:
            f.write(
                node.go_id
                + "\t"
                + node.name
                + "\t"
                + str(node.level)
                + "\t"
                + str(node.depth)
                + "\t"
                + str(len(node.annotations))
                + "\t"
                + str(len([x for x in node.annotations if x.is_manual is True]))
                + "\n"
            )

    with open(output_dir / Path("cc_go_terms.tsv"), "w") as f:
        f.write("Term\tName\tLevel\tDepth\tAnnotations\tManual Annotations\n")
        for node in cc_nodes:
            f.write(
                node.go_id
                + "\t"
                + node.name
                + "\t"
                + str(node.level)
                + "\t"
                + str(node.depth)
                + "\t"
                + str(len(node.annotations))
                + "\t"
                + str(len([x for x in node.annotations if x.is_manual is True]))
                + "\n"
            )

    with open(output_dir / Path("mf_go_terms.tsv"), "w") as f:
        f.write("Term\tName\tLevel\tDepth\tAnnotations\tManual Annotations\n")
        for node in mf_nodes:
            f.write(
                node.go_id
                + "\t"
                + node.name
                + "\t"
                + str(node.level)
                + "\t"
                + str(node.depth)
                + "\t"
                + str(len(node.annotations))
                + "\t"
                + str(len([x for x in node.annotations if x.is_manual is True]))
                + "\n"
            )
    with open(output_dir / Path("go_terms_not_found.txt"), "w") as f:
        f.write("\n".join(dataset.go_terms_not_found))

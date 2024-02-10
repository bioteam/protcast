from protcast.preprocessing.simple_dataset import SimpleDataset
from protcast.globals import CC,BP,MF
from pathlib import Path
from typeguard import typechecked
import pandas as pd
import plotly.express as px

@typechecked
def create_stats_files(
    dataset_location: str
):
    """generate_dataset_stats
    Create simpledataset_statistics.txt, cc_go_terms.txt, bp_go_terms.txt,
    mf_go_terms.txt files and histograms of terms and levels.

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

        bp_annots = dataset.get_all_annotations(namespace=BP)
        cc_annots = dataset.get_all_annotations(namespace=CC)
        mf_annots = dataset.get_all_annotations(namespace=MF)

        f.write(f"Annotations in {BP}: {len(bp_annots)}\n")
        f.write(f"Annotations in {CC}: {len(cc_annots)}\n")
        f.write(f"Annotations in {MF}: {len(mf_annots)}\n\n")

        for level in range(14):
            bp_levels.append(len([x for x in bp_nodes if x.level == level]))
        for level in range(14):
            cc_levels.append(len([x for x in cc_nodes if x.level == level]))
        for level in range(14):
            mf_levels.append(len([x for x in mf_nodes if x.level == level]))

        f.write("Nodes by level (0-13)\n")
        f.write(BP + "\t" + "\t".join([str(x) for x in bp_levels]) + "\n")
        f.write(CC + "\t" + "\t".join([str(x) for x in cc_levels]) + "\n")
        f.write(MF + "\t" + "\t".join([str(x) for x in mf_levels]) + "\n")

    df = pd.DataFrame({"BP":bp_levels, "CC":cc_levels, "MF":mf_levels})
    fig = px.bar(df, x=df.index, y=["BP", "CC", "MF"], barmode="stack",
             title="GO Terms by Level", text_auto=True)
    fig.update_layout(xaxis_title="Level", yaxis_title="Number of Terms")
    fig.show()
    fig.write_image(output_dir/"GO_terms_by_level.png")

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

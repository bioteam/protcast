import os
import logging
import pickle

from pathlib import Path
from tqdm import tqdm
from typeguard import typechecked
from datetime import datetime
from Bio.SeqIO.FastaIO import FastaIterator

from protcast.preprocessing.parse_swissprot import parse_swissprot
from protcast.preprocessing.parse_gaf import parse_gaf
from protcast.preprocessing.annotation import Annotation
from protcast.preprocessing.annotated_godag import AnnotatedGODag
from protcast.preprocessing.annotated_godag import AnnotatedGOTerm
from protcast.preprocessing.protein import Protein
from protcast.preprocessing.utils import md5
from protcast.globals import CC, BP, MF


class SimpleDataset:
    """SimpleDataset
    This class runs the SwissProt, TrEMBL, GAF, and Gene Ontology file
    parsers and creates a SimpleDataset can be saved to disk.

    Attributes
    ----------
    proteins: dict
        Key is id and value is a Protein
    annotated_dag: AnnotatedDag
        The GO DAGs plus Annotations assigned to AnnotatedGOTerms
    go_terms_not_found: list
        ...
    ontology_path: Path
        Input file
    swissprot_path: Path
        Input file
    trembl_path: Path
        Input file
    gaf_path: Path
        Input file
    output_dir: Path
        Location of serialized SimpleDataset and log file
    created_at: Datetime
        ...
    verbose: bool
        Default is False.
    ontology_md5: str
        Checksum
    swissprot_md5: str
        Checksum

    Methods
    -------
    init:
        Initialize
    save:
        Save SimpleDataset to disk
    from_serialized_file:
        Load SimpleDataset from disk
    get_term: str
        Returns an AnnotatedGOTerm given an id
    get_all_annotations: None or namespace
        Returns a list of Annotations
    get_all_terms: None or namespace
        Returns a list of AnnotatedGOTerms
    add_proteins: dict
        Adds one or more Proteins
    get_annotatations_from_gaf:
        Get all Annotations from a *gaf file
    parse_fasta:
        Inner function, parse TrEMBL
    create_annotation_files
        ...
    remove_protein:
        ...
    md5:
        Get MD5 checksums
    to_obo:
        Write an OBO Flat file
    """

    @typechecked
    def __init__(
        self,
        ontology_path: Path,
        swissprot_path: Path,
        trembl_path: Path,
        gaf_path: Path,
        output_dir: Path,
        verbose: bool = False,
    ):
        """__init__
        ...

        Parameters
        ----------
        ontology_path: Path
            ...
        swissprot_path: Path
            ...
        trembl_path: Path
            ...
        gaf_path: Path
            ...
        output_dir: Path
            Location for saved SimpleDataset and log file
        verbose: bool
            Write DEBUG level log if True. Default is False.

        Returns
        -------
        None
        """
        self.verbose = verbose
        self.output_dir = output_dir
        self.created_at = datetime.now()
        self.ontology_path = ontology_path
        self.ontology_md5 = md5(self.ontology_path)
        self.swissprot_path = swissprot_path
        self.swissprot_md5 = md5(self.swissprot_path)
        # Due to size we skip md5 of the trembl and GOA *gaf files
        self.gaf_path = gaf_path
        self.trembl_path = trembl_path

        self.proteins = dict()
        self.accessions = dict()
        self.go_terms_not_found = set()

        logger = logging.getLogger()
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        self.write_log(logger)

        # Create AnnotatedGODag
        self.annotated_dag: AnnotatedGODag = AnnotatedGODag(self.ontology_path)
        # Parse proteins from SwissProt
        (
            uniprot_annotations,
            uniprot_proteins,
        ) = parse_swissprot(self.swissprot_path)

        self.add_proteins(uniprot_proteins)

        # Add UniProt Annotations to the annotated DAG
        num_added = self.add_annotations(uniprot_annotations)
        logging.info(f"{num_added} Annotations added from '{self.swissprot_path}'")

        # Parse annotations from UniProt-GOA *gaf file
        gaf_annotations, new_protein_ids = self.get_annotations_from_gaf()

        # Retrieve Proteins from Trembl for new protein ids from the GOA file
        new_proteins = self.parse_fasta(new_protein_ids)
        self.add_proteins(new_proteins)

        # Add Annotations found in UniProt-GOA *gaf file
        # but first verify that the Protein actually exists
        num_added = self.add_annotations(gaf_annotations, check_pid=True)
        logging.info(f"Added {num_added} Annotations from '{self.gaf_path}'")

        logging.info(f"GO: '{self.ontology_path}'")
        logging.info(f"GOA: '{self.gaf_path}'")
        logging.info(f"UniProt: '{self.swissprot_path}'")
        logging.info(f"TrEMBL: '{self.trembl_path}'")
        logging.info(f"Saved SimpleDataset: '{self.output_dir}'")
        logging.info(f"Created at: '{self.created_at}'")
        logging.debug(f"GO terms not found: '{self.go_terms_not_found}'")

    @typechecked
    def save(self) -> None:
        """save
        Saves SimpleDataset to disk

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        os.makedirs(self.output_dir, exist_ok=True)
        logging.debug(f"Saving SimpleDataset.bin to '{self.output_dir}'")
        with open(self.output_dir / Path("SimpleDataset.bin"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_serialized_file(cls, file: str):
        """from_serialized_file
        Gets a saved SimpleDataset from disk

        Parameters
        ----------
        file: str
            File location

        Returns
        -------
        SimpleDataset
        """
        with open(file, "rb") as f:
            ds = pickle.load(f)
        return ds

    def create_annotation_files(
        self, output_path: str, include_electronic: bool = False
    ):
        """create_annotation_files
        Creates *tsv files for all the proteins in each namespace and a
        text file with the ids of all missing proteins.

        Parameters
        ----------
        output_path: str
            ...
        include_electronic: bool
            Default is False

        Returns
        -------
        None
        """
        bp_output_path = output_path + "_bpo.tsv"
        cc_output_path = output_path + "_cco.tsv"
        mf_output_path = output_path + "_mfo.tsv"
        missing_proteins_path = output_path + "_missing_proteins.txt"

        bp_file = open(bp_output_path, "w")
        cc_file = open(cc_output_path, "w")
        mf_file = open(mf_output_path, "w")
        missing_proteins_file = open(missing_proteins_path, "w")

        namespace_file_map = {
            BP: bp_file,
            CC: cc_file,
            MF: mf_file,
        }

        for go_id in self.annotated_dag.parent.keys():
            go_term = self.annotated_dag.get_term(go_id)
            for annot in go_term.get_all_annotations():
                namespace_file_map[go_term.namespace].write(
                    annot.protein_id
                    + "\t"
                    + annot.go_id
                    + "\t"
                    + annot.evidence_code
                    + "\t"
                    + annot.is_manual
                    + "\n"
                )

        for protein in self.missing_proteins:
            missing_proteins_file.write(protein + "\n")

        bp_file.close()
        cc_file.close()
        mf_file.close()
        missing_proteins_file.close()

    def add_annotations(self, annotations, check_pid=False) -> int:
        """add_annotations
        ...

        Parameters
        ----------
        annotations: list
            list of Annotations

        Returns
        -------
        num_new_annotations: number of Annotations added
        """
        num_new_annotations = 0
        for annot in annotations:
            if check_pid:
                # There is no Protein for this Annotation
                if not self.accessions.get(annot.protein_id):
                    logging.debug(
                        f"No Protein found for {annot.protein_id} cannot add Annotation"
                    )
                    continue
            go_term = self.get_term(annot.go_id)
            if go_term:
                result = go_term.add_annotation(annot)
                if result:
                    num_new_annotations += 1
            else:
                self.go_terms_not_found.add(annot.go_id)
        return num_new_annotations

    def get_annotations_from_gaf(self):
        """get_annotations_from_gaf
        Read annotations from a *gaf file and return a list of Annotations and
        a list of protein ids that are not found in the UniProt input file.

        Parameters
        ----------
        None

        Returns
        -------
        gaf_annotations: list of Annotations
        new_protein_ids: list of protein ids
        """
        gaf_annotations = list()
        new_protein_ids = set()

        logging.debug(f"Reading from '{str(self.gaf_path)}'")
        gaf_lines = parse_gaf(self.gaf_path)

        for rec in tqdm(
            gaf_lines, desc=f"Processing GOA records from '{str(self.gaf_path)}'"
        ):
            gaf_annotations.append(
                Annotation(rec["DB_Object_ID"], rec["Evidence"], rec["GO_ID"])
            )

            # If the protein is not in SwissProt then it is a new protein
            if rec["DB_Object_ID"] not in self.accessions:
                logging.debug(f"New protein id: '{rec['DB_Object_ID']}'")
                new_protein_ids.add(rec["DB_Object_ID"])

        logging.info(
            f"Found {len(new_protein_ids)} new protein ids in '{self.gaf_path}'"
        )
        logging.info(f"Found {len(gaf_annotations)} Annotations in '{self.gaf_path}'")
        return gaf_annotations, new_protein_ids

    def parse_fasta(self, new_protein_ids: list) -> dict:
        """parse_fasta
        Returns proteins in TrEMBL that were in the *gaf file but not
        in the UniProt *dat file.

        Parameters
        ----------
        new_protein_ids: list
            list of ids

        Returns
        -------
        Dict of protein ids and Proteins
        """
        logging.debug("Adding proteins and annotations from TrEMBL")
        new_proteins = dict()

        trembl_handle = open(self.trembl_path, "r")
        for record in tqdm(
            FastaIterator(trembl_handle),
            desc=f"Reading TrEMBL records from '{trembl_handle.name}'",
        ):
            # >tr|A0A1D8RA60|A0A1D8RA60_9ARCH
            pids = record.id.split("|")
            if pids[1] in new_protein_ids:
                protein = Protein(pids[1], str(record.seq), [pids[1], pids[2]])
                logging.debug(f"Created new Protein {pids[1]} using TrEMBL")
                new_proteins[pids[1]] = protein
            elif pids[2] in new_protein_ids:
                protein = Protein(pids[2], str(record.seq), [pids[1], pids[2]])
                logging.debug(f"Created new Protein {pids[2]} using TrEMBL")
                new_proteins[pids[2]] = protein

        logging.info(
            f"Found {len(new_proteins.keys())} Proteins from '{self.gaf_path}' in TrEMBL"
        )
        return new_proteins

    def add_proteins(self, new_proteins: dict) -> None:
        """add_proteins
        Adds Proteins and populates the accessions dict

        Parameters
        ----------
        proteins: dict
            Key is id, value is Protein

        Returns
        -------
        None
        """
        self.proteins.update(new_proteins)
        for protein in new_proteins.values():
            for acc in protein.accessions:
                self.accessions[protein.id] = acc

    def remove_protein(self, protein_id: str) -> None:
        """remove_protein
        Deletes protein from SimpleDataset.proteins

        Parameters
        ----------
        protein_id: str
            Protein id

        Returns
        -------
        None
        """
        assert self.proteins[protein_id] is not None
        del self.proteins[protein_id]

    def write_log(self, logger) -> None:
        """write_log
        Write a log in the output directory

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        os.makedirs(self.output_dir, exist_ok=True)
        formatter = logging.Formatter("%(levelname)s | %(message)s")
        file_handler = logging.FileHandler(self.output_dir / "SimpleDataset.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def to_obo(self) -> None:
        """to_obo
        Write an OBO Flat file for all primary and non-obsolete GO terms.
        The protein identifiers from all the are written as *xref* values.

        [Term]
        id: GO:0000049
        name: tRNA binding
        namespace: molecular_function
        comment: level 4
        is_a: GO:0003723
        xref: P99087
        xref: Q23399

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Keys are GO ids, values is a list of Annotations using that GO term
        all_go_terms = {}
        for protein in self.proteins.values():
            for annot in protein.get_all_annotations():
                if annot.go_id not in all_go_terms:
                    all_go_terms[annot.go_id] = set()
                all_go_terms[annot.go_id].add(annot.protein_id)

        obo_output_path = self.output_dir / "SimpleDataset.obo"

        with open(obo_output_path, "w") as obo_file:
            obo_file.write(
                "format-version: 1.2\ndefault-namespace: gene_ontology\nontology: go\n\n"
            )
            for go_term_id in all_go_terms.keys():
                term = self.get_term(go_term_id)
                if term:
                    obo_file.write(
                        "[Term]"
                        + "\nid: "
                        + term.go_id
                        + "\nname: "
                        + term.name
                        + "\nnamespace: "
                        + term.namespace
                        + "\n"
                        + "\ncomment: level "
                        + str(term.level)
                        + "\n"
                    )
                    for parent in term.parents:
                        obo_file.write("is_a: " + parent + "\n")
                for protein_id in all_go_terms[go_term_id]:
                    obo_file.write("xref: " + protein_id + "\n")
                obo_file.write("\n")

    @typechecked
    def get_term(self, go_id: str) -> AnnotatedGOTerm | None:
        """get_term
        Get AnnotatedGOterm given an id

        Parameters
        ----------
        go_id: str
            Id of GOTerm

        Returns
        -------
        AnnotatedGOTerm or None
        """
        return self.annotated_dag.go_terms_map.get(go_id)

    @typechecked
    def get_all_terms(self, namespace=None) -> list[AnnotatedGOTerm]:
        """get_all_terms
        Get all AnnotatedGOterms

        Parameters
        ----------
        None or namespace

        Returns
        -------
        List of AnnotatedGOTerms
        """
        terms = list()
        for term in self.annotated_dag.go_terms_map.values():
            if namespace:
                if term.namespace == namespace:
                    terms.append(term)
            else:
                terms.append(term)
        return terms

    @typechecked
    def get_all_annotations(self, namespace=None) -> list:
        """get_all_annotations
        Get all Annotations

        Parameters
        ----------
        None or namespace

        Returns
        -------
        List of Annotations
        """
        annots = list()
        for term in self.annotated_dag.go_terms_map.values():
            if namespace:
                if term.namespace == namespace:
                    annots.extend(term.annotations)
            else:
                annots.extend(term.annotations)
        return annots

    @typechecked
    def get_descendants(self, go_id: str) -> list[str]:
        """
        Recursively retrieve all the GO ids of all descendants of a GO term

        Parameters
        ----------
        go_id: str
            GO ID of the starting node

        Returns
        -------
        A list of all descendant GO IDs
        """
        descendants = []
        if go_id in self.annotated_dag.go_terms_map:
            for child_id in self.annotated_dag.go_terms_map[go_id].children:
                descendants.append(child_id)
                descendants.extend(self.get_descendants(child_id))
        return descendants

from __future__ import annotations

import os

from datetime import datetime
import hashlib
import logging
from pathlib import Path
import pickle
from tqdm import tqdm
from typeguard import typechecked

from Bio.Seq import Seq
from Bio.SeqIO.FastaIO import FastaIterator

from protcast import BP, CC, MF
from protcast.preprocessing.annotation import Annotation
from protcast.preprocessing.ontology import Ontology
from protcast.preprocessing.protein import Protein

from preprocessing.parse_swissprot import parse_swissprot
from preprocessing.parse_gaf import parse_gaf


class SimpleDataset:
    """SimpleDataset
    This class runs the SwissProt, TrEMBL, GAF, and Gene Ontology file
    parsers and creates a SimpleDataset can be saved to disk. Used by
    *preprocessing/create_simple_dataset.py*.

    Attributes
    ----------
    proteins: dict
        ....
    accessions: dict
        ...
    go_terms_not_found: list
        ...
    ontology_path: Path
        ...
    swissprot_path: Path
        ...
    trembl_path: Path
        ...
    gaf_path: Path
        ...
    output_dir: Path
        ...
    verbose: bool
        ...
    ontology_md5: str
        ...
    swissprot_md5: str
        ...

    Methods
    -------
    init:
        Initialize
    save:
        Save SimpleDataset to disk
    from_serialized_file:
        Load SimpleDataset from disk
    create_annotation_files
        ...
    _propagate_annotations
        ...
    _annotate_proteins_from_gaf:
        ...
    _add_trembl_proteins:
        ...
    _parse:
        Inner function, parse TrEMBL
    remove_protein:
        ...
    md5:
        Get MD5 checksums
    """

    @typechecked
    def __init__(
        self,
        ontology_path: Path,
        swissprot_path: Path,
        trembl_path: Path,
        gaf_path: Path,
        output_dir: Path,
        verbose: bool,
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
            Write DEBUG level log if True

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

        logger = logging.getLogger()
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        self._write_log(logger)

        # Keep track of proteins seen in the *gaf file but
        # not found in Uniprot or TrEMBL
        self.missing_proteins = set()

        # Create Ontology
        self.ontology: Ontology = Ontology(self.ontology_path)
        # Parse proteins from SwissProt
        (
            self.proteins,
            self.go_terms_not_found,
            self.accessions,
        ) = parse_swissprot(self.ontology, self.swissprot_path)

        # Assert that all terms seen in UniProtKB/SwissProt are in the Ontology
        assert len(self.go_terms_not_found) == 0

        # Parse annotations from UniProt-GOA *gaf file
        trembl_annotations = self._annotate_proteins_from_gaf()

        # Add proteins found in UniProt-GOA *gaf file that are in TrEMBL
        self._add_trembl_proteins(trembl_annotations)

        #
        self._propagate_annotations()

        logging.info(f"GO: '{self.ontology_path}'")
        logging.info(f"GOA: '{self.gaf_path}'")
        logging.info(f"UniProt: '{self.swissprot_path}'")
        logging.info(f"TrEMBL: '{self.trembl_path}'")
        logging.info(f"Saved SimpleDataset: '{self.output_dir}'")
        logging.info(f"Created at: '{self.created_at}'")

    @typechecked
    def save(self):
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
    def from_serialized_file(cls, file: str) -> SimpleDataset:
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

        for protein in self.proteins.values():
            protein_annots = protein.get_manual_annotations()
            if include_electronic:
                protein_annots.extend(protein.get_electronic_annotations())
            for annotation in protein_annots:
                go_term = self.ontology.get_primary_term(annotation.go_term_id)
                namespace_file_map[go_term.namespace].write(
                    protein.id + "\t" + annotation.go_term_id + "\n"
                )

        for protein in self.missing_proteins:
            missing_proteins_file.write(protein + "\n")

        bp_file.close()
        cc_file.close()
        mf_file.close()
        missing_proteins_file.close()

    def _propagate_annotations(self):
        """propagate_annotations
        Adds detailed information on the ancestors of the GO term in
        each Annotation including whether the ancestor 'has_obsolete'
        and 'is_manual'.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for protein in self.proteins.values():
            logging.debug(
                f"Propagating {len(protein.get_all_annotations())} annotations for {protein.id}"
            )
            protein_annots = list()
            for annot in protein.get_all_annotations():
                protein_annots.append(annot.go_term_id)
                # The annotation key is the GO term id
                go_term_ancestors: list[str] = self.ontology.get_primary_term(
                    annot.go_term_id
                ).ancestors
                for ancestor in go_term_ancestors:
                    ancestor_annot = protein.get_annotation(ancestor)
                    if not ancestor_annot:
                        ancestor_go_term = self.ontology.get_primary_term(
                            ancestor
                        )
                        ancestor_annot = Annotation(
                            ancestor_go_term.id,
                            protein.id,
                            annot.evidence_code,
                            ancestor_go_term,
                        )
                        protein.add_annotation(ancestor_annot)
                    else:
                        # If the protein is already annotated with this GO
                        # term, we might update 'is_manual'
                        ancestor_annot.is_manual = annot.is_manual

    @typechecked
    def _annotate_proteins_from_gaf(self) -> list[tuple]:
        """annotate_proteins_from_gaf
        Adds Annotations to the SimpleDataset.ontology and returns the
        annotations that belong to the UniProtKB/TrEMBL database but do not
        have DB_Object_IDs. It is a list of tuples:

        (protein ID, GO term, evidence code)

        Parameters
        ----------
        None

        Returns
        -------
        trembl_annotations: list of tuples
        """
        trembl_annotations = []
        num_swissprot_annots = 0
        num_new_swissprot_annots = 0
        num_annotations_not_labeled_uniprotkb = 0
        num_annotations_labeled_uniprotkb = 0

        logging.debug(f"Reading from '{str(self.gaf_path)}'")
        gaf_annotations = parse_gaf(self.gaf_path)

        for rec in tqdm(
            gaf_annotations,
            desc=f"Processing GOA records from '{str(self.gaf_path)}'",
        ):
            go_term = self.ontology.get_primary_term(rec["GO_ID"])

            # Not "UniProtKB" but can try to get a protein sequence from TrEMBL
            if rec["DB"] != "UniProtKB":
                logging.debug(
                    f"Found protein {rec['DB_Object_ID']} labeled '{rec['DB']} not 'UniProtKB'"
                )
                num_annotations_not_labeled_uniprotkb += 1
                trembl_annotations.append(
                    (
                        rec["DB_Object_ID"],
                        rec["GO_ID"],
                        rec["Evidence"],
                        go_term,
                    )
                )
                continue

            num_annotations_labeled_uniprotkb += 1

            # See if the protein is in SwissProt
            if self.accessions.get(rec["DB_Object_ID"]):
                primary_accession = self.accessions[rec["DB_Object_ID"]]
                # Already have Protein and Annotation
                if self._find_annotation(rec, primary_accession):
                    num_swissprot_annots += 1
                # Do not have the Annotation so add it to the Protein
                else:
                    annot = Annotation(
                        rec["GO_ID"],
                        primary_accession,
                        rec["Evidence"],
                        go_term,
                    )
                    self.proteins[primary_accession].add_annotation(annot)
                    logging.debug(
                        f"Created new annotation for {primary_accession}: {rec['GO_ID']}, {rec['Evidence']}"
                    )
                    num_new_swissprot_annots += 1
            # The GAF DB_Object_ID does not match any accession from SwissProt
            else:
                logging.debug(
                    "No accession found in SwissProt for 'UniProtKB' protein "
                    f"{rec['DB_Object_ID']} "
                )
                trembl_annotations.append(
                    (
                        rec["DB_Object_ID"],
                        rec["GO_ID"],
                        rec["Evidence"],
                        go_term.is_obsolete,
                    )
                )

        logging.info(
            f"Found {len(gaf_annotations)} total annotations in '{self.gaf_path}'"
        )
        logging.info(
            f"Found {num_annotations_not_labeled_uniprotkb} annotations not labelled 'UniProtKB'"
        )
        logging.info(
            f"Found {num_annotations_labeled_uniprotkb} annotations labelled 'UniProtKB'"
        )
        logging.info(
            f"Found {num_swissprot_annots} 'UniProtKB' annotations in Swissprot"
        )
        logging.info(
            f"Found {len(trembl_annotations)} 'UniProtKB' annotations not in SwissProt"
        )
        logging.info(
            f"Made {num_new_swissprot_annots} new Annotations for existing SwissProt Proteins"
        )

        return trembl_annotations

    def _find_annotation(self, rec, primary_accession):
        protein = self.proteins.get(primary_accession)
        for annot in protein.get_all_annotations():
            if (
                annot.evidence_code == rec["Evidence"]
                and annot.go_term_id == rec["GO_ID"]
            ):
                logging.debug(
                    f"Found Annotation for {primary_accession} ({rec['GO_ID']}, {rec['Evidence']})"
                )
                return True
        return False

    @typechecked
    def _add_trembl_proteins(self, trembl_annotations: list[tuple]) -> None:
        """_add_trembl_proteins
        Find proteins in TrEMBL that were in the *gaf file but not
        in the UniProt *dat file.

        Parameters
        ----------
        trembl_annotations:
            ...

        Returns
        -------
        None
        """
        logging.debug("Adding proteins and annotations from TrEMBL")

        with open(self.trembl_path, "r") as trembl_handle:
            self._parse_trembl(trembl_handle, trembl_annotations)

    def _parse_trembl(self, trembl_handle, trembl_annotations):
        # We parse both TrEMBL and SwissProt because there are
        # entries in Uniprot-GOA *gaf file that can be missing from Swissprot.
        # These databases are not completely in sync at any point in time.
        annotated_trembl_ids = set([x[0] for x in trembl_annotations])
        annotated_trembl_seqs = {}
        new_proteins_from_trembl = 0
        new_annotations_from_trembl = 0

        for record in tqdm(
            FastaIterator(trembl_handle),
            desc=f"Reading TrEMBL records from '{trembl_handle.name}'",
        ):
            # >tr|A0A1D8RA60|A0A1D8RA60_9ARCH
            accession = record.id.split("|")[1]
            if accession in annotated_trembl_ids:
                self.accessions[accession] = accession
                annotated_trembl_seqs[accession] = str(record.seq)

        for annot in tqdm(
            trembl_annotations,
            desc=f"Processing records from '{trembl_handle.name}'",
        ):
            protein_id, go_term_id, evidence, has_obsolete = annot
            go_term = self.ontology.get_primary_term(go_term_id)

            primary_accession = self.accessions.get(protein_id)
            sequence = annotated_trembl_seqs.get(primary_accession)
            if sequence is None:
                logging.debug(
                    f"Could not find protein {protein_id} in TrEMBL. Skipping."
                )
                self.missing_proteins.add(protein_id)
                continue

            # Make a new Protein if it does not exist
            protein = self.proteins.get(primary_accession)
            if protein is None:
                protein = Protein(
                    primary_accession, annotated_trembl_seqs[primary_accession]
                )
                new_proteins_from_trembl += 1
                logging.debug(f"Created new Protein {protein_id} using TrEMBL")

                # The annotation can already exist due to the fact that the
                # UniProt-GOA *gaf file can have almost identical entries that
                # contain the same 'DB_Object_ID' and 'GO_ID' effectively
                # yielding the same annotation. For example:
                #
                # UniProtKB	A0A093IGM8	N326_07474	involved_in	GO:0032099	GO_REF:0000024	ISS	UniProtKB:P68259
                # UniProtKB	A0A093IGM8	N326_07474	involved_in	GO:0032099	GO_REF:0000024	ISS	UniProtKB:Q3HWX0
            logging.debug(f"Protein {protein.id} found in TrEMBL")

            annotation = Annotation(
                go_term_id,
                primary_accession,
                evidence,
                go_term,
            )
            protein.add_annotation(annotation)
            new_annotations_from_trembl += 1
            self.proteins[protein.id] = protein
            logging.debug(
                f"Added Annotation ({go_term_id}, {evidence}) to Protein {protein.id} from TrEMBL"
            )

        logging.info(
            f"Made {new_proteins_from_trembl} new Proteins from TrEMBL"
        )
        logging.info(
            f"Made {new_annotations_from_trembl} new Annotations from TrEMBL"
        )

    def remove_protein(self, protein_id: str) -> None:
        """remove_protein
        Deletes protein from Dataset.proteins

        Parameters
        ----------
        protein_id: str
            ...

        Returns
        -------
        None
        """
        assert self.proteins[protein_id] is not None
        del self.proteins[protein_id]

    def _write_log(self, logger) -> None:
        """_write_log
        Write a log in the output directory

        Args:
            None

        Returns:
            None
        """
        os.makedirs(self.output_dir, exist_ok=True)
        formatter = logging.Formatter("%(levelname)s | %(message)s")
        file_handler = logging.FileHandler(
            self.output_dir / "SimpleDataset.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def md5(file_path: str) -> str:
    """md5
    Calculate the md5 hash of a file.

    Args:
        file_path (str): The path to the file to be hashed.

    Returns:
        str: The md5 hash as a string.
    """
    logging.info(f"Calculating md5 of {str(file_path)}")

    with open(file_path, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()

    return file_hash

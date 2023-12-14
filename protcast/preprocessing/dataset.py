from __future__ import annotations

import os

from datetime import datetime
import hashlib
import logging
from pathlib import Path
import pickle
from tqdm import tqdm
from typeguard import typechecked

from Bio import SwissProt
from Bio.Seq import Seq
from Bio.SeqIO.FastaIO import FastaIterator
from Bio.UniProt.GOA import gafiterator

from protcast import BP, CC, MF
from protcast.preprocessing.annotation import Annotation
from protcast.preprocessing.ontology import Ontology
from protcast.preprocessing.protein import Protein

from preprocessing.parse_swissprot import parse_swissprot


class Dataset:
    """Dataset
    This class runs the SwissProt, TrEMBL, UniProtGOA and Gene Ontology file
    parsers and creates a single dataset which is saved to disk. Used by
    *preprocessing-scripts/create_dataset.py*.

    Attributes
    ----------
    name: type
        ....

    Methods
    -------
    init:
        Initialize
    ...:
        ...
    """

    @typechecked
    def __init__(
        self,
        ontology_path: Path,
        swissprot_t0_path: Path,
        swissprot_t1_path: Path,
        trembl_path: Path,
        goa_path: Path,
    ):
        """__init__
        ...

        Parameters
        ----------
        ontology_paty: Path
            ...
        swissprot_t0_paty: Path
            ...
        swissprot_t1_paty: Path
            ...
        trembl_paty: Path
            ...
        goa_paty: Path
            ...

        Returns
        -------
        None
        """
        self.created_at = datetime.now()
        self.ontology_path = ontology_path
        self.ontology_md5 = md5(self.ontology_path)
        self.swissprot_t0_path = swissprot_t0_path
        self.swissprot_t0_md5 = md5(self.swissprot_t0_path)
        self.swissprot_t1_path = swissprot_t1_path
        self.swissprot_t1_md5 = md5(self.swissprot_t1_path)
        self.trembl_path = trembl_path
        # Due to size we skip md5 of UniProtKB/TrEMBL
        self.goa_path = goa_path
        self.goa_md5 = md5(self.goa_path)

        # Keeps track of proteins seen in Uniprot-GOA annotations but not found
        # in UniprotKB/{SwissProt / TrEMBL}
        self.missing_proteins = set()

        # Create Ontology
        self.ontology: Ontology = Ontology(ontology_path)

        # Parse proteins from SwissProt
        (
            self.proteins,
            self.go_terms_not_found,
            self.accessions,
        ) = parse_swissprot(self.ontology, self.swissprot_t0_path)
        # Assert that all terms seen in UniProtKB/SwissProt are in the Ontology
        assert len(self.go_terms_not_found) == 0

        # Parse annotations from UniProt-GOA
        trembl_annotations = self.annotate_proteins_from_goa(
            self.proteins, self.goa_path
        )

        # Add proteins found in UniProt-GOA that are in TrEMBL
        self.add_trembl_proteins(
            self.swissprot_t1_path,
            self.trembl_path,
            trembl_annotations,
        )

        self.propagate_annotations()

    @typechecked
    def save(self, output_dir: Path):
        """save
        Saves single dataset to disk

        Parameters
        ----------
        output_dir: Path
            ...

        Returns
        -------
        None
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / Path("dataset.bin"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_serialized_file(cls, file: str) -> Dataset:
        """from_serialized_file
        Gets a saved Dataset from disk

        Parameters
        ----------
        file: str
            ...

        Returns
        -------
        Dataset
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
        include_electronic: Boolean
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
            protein_annots = (
                protein.get_manual_non_obsolete_annotations()
            )
            if include_electronic:
                protein_annots.extend(
                    protein.get_electronic_non_obsolete_annotations()
                )
            for annotation in protein_annots:
                go_term = self.ontology.get_primary_term(
                    annotation.go_term_id
                )
                namespace_file_map[go_term.namespace].write(
                    protein.id + "\t" + annotation.go_term_id + "\n"
                )

        for protein in self.missing_proteins:
            missing_proteins_file.write(protein + "\n")

        bp_file.close()
        cc_file.close()
        mf_file.close()
        missing_proteins_file.close()

    def propagate_annotations(self):
        """propagate_annotations
        Adds detailed information on the ancestors of the GO term in
        each Annotation including whether the ancestor 'is_obsolete'
        and 'is_manual'.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for protein in self.proteins.values():
            for annot in protein.get_all_annotations():
                # The annotation key is the GO term id
                go_term_ancestors: list[
                    str
                ] = self.ontology.get_primary_term(
                    annot.go_term_id
                ).ancestors
                for ancestor in go_term_ancestors:
                    ancestor_annot = protein.get_annotation(ancestor)
                    if not ancestor_annot:
                        ancestor_go_term = (
                            self.ontology.get_primary_term(ancestor)
                        )
                        ancestor_annot = Annotation(
                            ancestor_go_term.id,
                            protein.id,
                            False,
                            ancestor_go_term.is_obsolete,
                            is_manual=annot.is_manual,
                        )
                        protein.add_annotation(ancestor_annot)
                        ancestor_go_term.add_annotation(
                            ancestor_annot
                        )
                    else:
                        # If the protein is already annotated with this GO
                        # term, we might update 'is_manual'.
                        ancestor_annot.set_is_manual(
                            ancestor_annot.is_manual
                            or annot.is_manual
                        )

    @typechecked
    def annotate_proteins_from_goa(
        self, proteins: dict[str, Protein], goa_path: Path
    ) -> list[tuple]:
        """annotate_proteins_from_goa
        Adds Annotations to the Dataset.ontology and returns the annotations
        that belong to the UniProtKB/TrEMBL database but do not have DB_Object_IDs.
        It is a list of tuples: (protein ID, GO term, evidence cpode)

        Parameters
        ----------
        proteins: dict
            ...
        goa_path: Path
            ...

        Returns
        -------
        trembl_annotations: list of tuples
        """
        trembl_annotations = []
        swissprot_annotations = 0
        not_in_swissprot_annotations = 0

        with open(goa_path, "r") as handle:
            for rec in tqdm(
                gafiterator(handle),
                desc="Reading GAF records from {}".format(
                    str(goa_path)
                ),
            ):
                if rec["DB"] == "UniProtKB":
                    primary_accession_id = self.accessions.get(
                        rec["DB_Object_ID"]
                    )
                    primary_go_term = self.ontology.get_primary_term(
                        rec["GO_ID"]
                    )
                    if primary_accession_id:
                        if (
                            primary_accession_id
                            != rec["DB_Object_ID"]
                        ):
                            logging.debug(
                                "Found secondary protein id: "
                                f"{rec['DB_Object_ID']}. Primary is: "
                                f"{primary_accession_id}"
                            )

                        protein: Protein = proteins.get(
                            primary_accession_id
                        )
                        annot: Annotation = protein.get_annotation(
                            primary_go_term.id
                        )
                        # The annotation can already exist because the
                        # it can already be present in the UniProtKB/Swiss-Prot
                        # database or due to the fact that the UniProt-GOA
                        # database can have almost identical entries that
                        # contain the same 'DB_Object_ID' and 'GO_ID'
                        if annot:
                            # The 'Evidence Code' field is required in GAF 2*
                            annot.set_is_manual(
                                annot.is_manual
                                or rec["Evidence"] != "IEA"
                            )
                            swissprot_annotations += 1
                        else:
                            annot = Annotation(
                                primary_go_term.id,
                                protein.id,
                                True,
                                primary_go_term.is_obsolete,
                                rec["Evidence"] != "IEA",
                            )
                            protein.add_annotation(annot)
                            self.ontology.get_primary_term(
                                primary_go_term.id
                            ).add_annotation(annot)
                            not_in_swissprot_annotations += 1
                    else:
                        logging.debug(
                            "No primary accession ID found for protein ID:"
                            f"{rec['DB_Object_ID']} and evidence: "
                            f"{rec['Evidence']}"
                        )
                        trembl_annotations.append(
                            (
                                rec["DB_Object_ID"],
                                primary_go_term.id,
                                rec["Evidence"],
                            )
                        )
                else:
                    logging.debug(
                        f"Found protein {rec['DB_Object_ID']} that belongs to "
                        f"{rec['DB']}"
                    )

        logging.info(
            f"Found {len(trembl_annotations)} TrEMBL annotations in the GOA "
            "database"
        )
        logging.info(
            f"There were {swissprot_annotations} annotations in GOA that were "
            "seen in UniProtKB/SwissProt"
        )
        logging.info(
            f"There were {not_in_swissprot_annotations} annotations in GOA "
            "that were NOT seen in UniProtKB/SwissProt"
        )

        return trembl_annotations

    @typechecked
    def add_trembl_proteins(
        self,
        swissprot_t1_path,
        trembl_path,
        trembl_annotations: list[tuple],
    ):
        """add_trembl_proteins
        ...

        Parameters
        ----------
        swissprot_t1_path:
            ...
        trembl_path:
            ...
        trembl_annotations:
            ...

        Returns
        -------
        None
        """

        def parse(
            trembl_handle, tr_path, swissprot_handle, swiss_path
        ):
            # We parse both TrEMBL and SwissProt at t1 because there are
            # entries in Uniprot-GOA that can be missing from Swissprot at t0.
            # These databases are not completely in sync at any point in time.
            for record in tqdm(
                FastaIterator(trembl_handle),
                desc="Reading TrEMBL records",
            ):
                accession = record.id.split("|")[1]
                if accession in annotated_protein_ids:
                    self.accessions[accession] = accession
                    annotated_protein_seqs[accession] = record.seq

            for req in tqdm(
                SwissProt.parse(swissprot_handle),
                desc="Reading t1 SwissProt records",
            ):
                primary_accession = req.accessions[0]
                for acc in req.accessions:
                    if acc in annotated_protein_ids:
                        self.accessions[acc] = primary_accession
                        annotated_protein_seqs[
                            primary_accession
                        ] = req.sequence

            for (
                protein_id,
                go_term_id,
                evidence,
            ) in trembl_annotations:
                primary_accession = self.accessions.get(protein_id)
                sequence = annotated_protein_seqs.get(
                    primary_accession
                )
                if sequence is None:
                    logging.warning(
                        f"Couldn't find protein {protein_id} in neither "
                        "SwissProt nor TrEMBL. Most likely this protein is "
                        "obsolete and/or has been moved to UniRef or UniParc. "
                        "Skipping it..."
                    )
                    self.missing_proteins.add(protein_id)
                    continue

                protein = self.proteins.get(primary_accession)
                if protein is None:
                    protein = Protein(
                        primary_accession, Seq(sequence)
                    )
                    self.proteins[protein.id] = protein

                # The annotation can already exist due to the fact that the
                # UniProt-GOA database can have almost identical entries that
                # contain the same 'DB_Object_ID' and 'GO_ID' effectively
                # yielding the same annotation. For example:
                #
                # UniProtKB	A0A093IGM8	N326_07474	involved_in	GO:0032099	GO_REF:0000024	ISS	UniProtKB:P68259
                # UniProtKB	A0A093IGM8	N326_07474	involved_in	GO:0032099	GO_REF:0000024	ISS	UniProtKB:Q3HWX0
                annot = protein.get_annotation(go_term_id)
                if not annot:
                    go_term = self.ontology.get_primary_term(
                        go_term_id
                    )
                    annotation = Annotation(
                        go_term_id,
                        primary_accession,
                        True,
                        go_term.is_obsolete,
                        evidence != "IEA",
                    )

                    protein.add_annotation(annotation)
                    go_term.add_annotation(annotation)
                else:
                    annot.set_is_manual(
                        annot.is_manual or evidence != "IEA"
                    )

        logging.info(
            "Adding proteins and annotations from UniProtKB/TrEMBL"
        )
        annotated_protein_ids = set(
            [x[0] for x in trembl_annotations]
        )
        annotated_protein_seqs = {}

        with open(trembl_path, "r") as trembl_handle:
            with open(swissprot_t1_path, "r") as swissprot_handle:
                parse(
                    trembl_handle,
                    trembl_path,
                    swissprot_handle,
                    swissprot_t1_path,
                )

    def remove_protein(self, protein_id: str):
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


def md5(file_path):
    """md5
    Calculates md5

    Parameters
    ----------
    file_path: str
        ...

    Returns
    -------
    String
    """
    logging.info(f"Calculating md5 of {file_path}")
    with open(file_path, "rb") as file_to_check:
        data = file_to_check.read()
        return hashlib.md5(data).hexdigest()

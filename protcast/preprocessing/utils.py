import logging
import hashlib
from pathlib import Path
from typeguard import typechecked
from tqdm import tqdm
from Bio.SeqIO.FastaIO import FastaIterator
from protcast.preprocessing.protein import Protein


@typechecked
def md5(file_path: Path) -> str:
    """md5
    Calculate the md5 hash of a file

    Parameters
    ----------
    file_path: Path
        The path to the file to be hashed.

    Returns
    -------
    str: The md5 hash as a string.
    """
    logging.debug(f"Calculating md5 of {str(file_path)}")

    with open(file_path, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()

    return file_hash


@typechecked
def parse_fasta(trembl_path, gaf_path, new_protein_ids: set) -> dict:
    """parse_fasta
    Returns proteins in TrEMBL that were in the *gaf file but not
    in the UniProt *dat file.

    Parameters
    ----------
    new_protein_ids: set
        list of ids

    Returns
    -------
    Dict of protein ids and Proteins
    """
    logging.debug("Adding proteins and annotations from TrEMBL")
    new_proteins = dict()

    trembl_handle = open(trembl_path, "r")
    for record in tqdm(
        FastaIterator(trembl_handle),
        desc=f"Reading TrEMBL records from '{trembl_handle.name}'",
    ):
        # >tr|A0A1D8RA60|A0A1D8RA60_9ARCH
        pids = record.id.split("|")
        if pids[1] in new_protein_ids:
            pids[0] = pids[1]
        elif pids[2] in new_protein_ids:
            pids[0] = pids[2]

        protein = Protein(pids[0], str(record.seq), [pids[1], pids[2]])
        logging.debug(f"Created new Protein {pids[0]} using TrEMBL")
        new_proteins[pids[0]] = protein

    logging.info(
        f"Found {len(new_proteins.keys())} Proteins from '{gaf_path}' in TrEMBL"
    )
    return new_proteins

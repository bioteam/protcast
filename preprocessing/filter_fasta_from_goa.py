import argparse
import logging
from tqdm import tqdm

from Bio.SeqIO.FastaIO import FastaIterator, FastaWriter
from Bio.UniProt.GOA import gafiterator


def main():
    """"filter_fasta_from_goa.py
    This script takes an UniProt-GOA database and UniProt FASTA file and
    outputs a trimmed FASTA file that contains the sequences only seen in
    UniProt-GOA. This script is helpful to generate a FASTA file to give as
    input to the script that generates the Dataset (for the trembl option).
    
    Example:

    python preprocessing-scripts/filter_fasta_from_goa.py \
    data/goa/t0/goa_uniprot_all_noiea.gaf \
    ~/gargantua/trembl/uniprot_trembl.fasta \
    data/trembl/trembl_seqs_found_in_goa_2021_10_26.fasta
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "goa", help="Path to Gene Ontology Annotation (.gaf file)"
    )
    parser.add_argument(
        "trembl", help="Path to TrEMBL DB (.fasta file)"
    )
    parser.add_argument("output", help="Output file")
    args = parser.parse_args()

    goa_path = args.goa
    trembl_path = args.trembl
    output_path = args.output

    protein_ids = set()
    saved_records = []
    with open(goa_path, "r") as uniprot_goa:
        for rec in tqdm(
            gafiterator(uniprot_goa),
            desc="Iterating through UniProtKB-GOA",
        ):
            protein_ids.add(rec["DB_Object_ID"])

    records_saved = 0
    with open(trembl_path) as trembl:
        for record in tqdm(
            FastaIterator(trembl),
            desc="Reading TrEMBL in {}".format(str(trembl_path)),
        ):
            # TODO: Account for the fact of secondary accessions. Looks like
            # there are none in FASTA as of now
            accession = record.id.split("|")[1]
            if accession in protein_ids:
                records_saved += 1
                logging.info(
                    f"Saving record {records_saved}: {record.id}"
                )
                saved_records.append(record)

    logging.info(f"Found {len(saved_records)} matching sequences")

    with open(output_path, "w") as output_file:
        writer = FastaWriter(output_file)
        writer.write_file(saved_records)


if __name__ == "__main__":
    main()

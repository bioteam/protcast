import sys

import argparse
import logging
import pickle

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.ontology import Ontology
from preprocessing.parse_swissprot import parse_swissprot


def main():
    """swissprot2csv.py
    This code makes a 2-D matrix of protein ids, protein sequences, GO,
    KEGG term occurences, and protein feature vectors. The number of terms
    will vary by sequence and this will be a very sparse matrix with some
    1's ("has term") and many 0's ("does not have term"). The name of the
    feature vector is in the header. Example mini-matrix:

    PID,Sequence,GO:123,GO:456,GO:789,vg:123,vg:456,paac,paac,paac
    022L_IIV3, MAKALTYYCCK, 0, 1, 0, 0, 0, 4.56, 0.0, 3.0
    022L_IIV4, MLYlTYMRGGLCGVDKR, 0, 0, 1, 0, 0, 2.56, 7.0, 0.0
    22L_IIV5, MAFYTWMCLVVL, 1, 0, 0, 0, 0, 0.56, 0.0, 0.0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("ontology", help="Path to ontology file")
    parser.add_argument("uniprot_db", help="Path to uniprot db")
    parser.add_argument(
        "-o", "--output", help="Output swissprot file"
    )
    parser.add_argument("-v", action="store_true", help="Verbose")
    args = parser.parse_args()

    if args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ontology = Ontology.load_ontology(args.ontology)

    proteins, go_terms_not_found, accessions = parse_swissprot(
        ontology, args.uniprot_db
    )

    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump(proteins, f)


if __name__ == "__main__":
    main()

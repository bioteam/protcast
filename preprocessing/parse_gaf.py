import logging
from tqdm import tqdm

from pathlib import Path

GAF20FIELDS = [
    "DB",
    "DB_Object_ID",
    "DB_Object_Symbol",
    "Qualifier",
    "GO_ID",
    "DB:Reference",
    "Evidence",
    "With",
    "Aspect",
    "DB_Object_Name",
    "Synonym",
    "DB_Object_Type",
    "Taxon_ID",
    "Date",
    "Assigned_By",
    "Annotation_Extension",
    "Gene_Product_Form_ID",
]


def parse_gaf(gaf_file: Path) -> list[dict[str, dict[str, str]]]:
    """parse_gaf
    Parses a version 2.0 or 2.2 GO Annotation Format (GAF) file and returns a list 
    containing dicts with the 17 GAF keys. 
    
    The BioPython parser requires a header but the file supplied by Gene Ontology 
    is missing this header as of 11/2023 so their parser no longer works.
    See https://biopython.org/docs/1.75/api/Bio.UniProt.GOA.html

    Example line:

    UniProtKB	A0A1Z4V764	NIES806_33130	involved_in	GO:0006508	GO_REF:0000043	\
    IEA	UniProtKB-KW:KW-0645	P	Uncharacterized protein	NIES806_33130	\
    protein	taxon:1973481	20211010	UniProt
    
    > cat filtered_goa_uniprot_all_noiea.gaf|grep -v '!'|cut -f1|sort|uniq
    ComplexPortal
    RNAcentral
    UniProtKB

    402, 5952, 310418 entries, respectively. RNAcentral is a non-coding RNA database.

    Parameters
    ----------
    gaf_file: Path
        Path to GAF file

    Returns
    -------
    List of dicts with the 17 GOA keys
    """
    goas = list()

    with open(gaf_file, "r") as h:
        for line in h:
            if line.startswith("!"):
                continue
            goas.append(_parse_line(line))
    return goas


def _parse_line(inline):
    inrec = inline.rstrip("\n").split("\t")
    inrec[3] = inrec[3].split("|")  # Qualifier
    inrec[5] = inrec[5].split("|")  # DB:reference(s)
    inrec[7] = inrec[7].split("|")  # With || From
    inrec[10] = inrec[10].split("|")  # Synonym
    inrec[12] = inrec[12].split("|")  # Taxon
    return dict(zip(GAF20FIELDS, inrec))

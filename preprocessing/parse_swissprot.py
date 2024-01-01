from protcast.preprocessing.annotation import Annotation
from protcast.preprocessing.ontology import Ontology
from protcast.preprocessing.protein import Protein
import gzip
import logging
from tqdm import tqdm
from typeguard import typechecked
from pathlib import Path
from Bio import SwissProt


@typechecked
def parse_swissprot(
    ontology: Ontology,
    swissprot_db: Path,
) -> tuple[dict[str, Protein], set[str], dict[str, str]]:
    """parse_swissprot

    Example Uniprot file:

    ID   022L_IIV3               Reviewed;         225 AA.
    AC   Q197D8;
    DT   11-JUL-2006, sequence version 1.
    DE   RecName: Full=Transmembrane protein 022L;
    GN   ORFNames=IIV3-022L;
    OS   Invertebrate iridescent virus 3 (IIV-3) (Mosquito iridescent virus).
    OC   Viruses; Iridoviridae; Betairidovirinae; Chloriridovirus.
    OX   NCBI_TaxID=345201;
    OH   NCBI_TaxID=7163; Aedes vexans (Inland floodwater mosquito) (Culex vexans).
    RN   [1]
    RP   NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA].
    RX   PubMed=16912294; DOI=10.1128/jvi.00464-06;
    RA   Delhon G., Tulman E.R., Afonso C.L., Lu Z., Becnel J.J., Moser B.A.,
    RA   Kutish G.F., Rock D.L.;
    RT   "Genome of invertebrate iridescent virus type 3 (mosquito iridescent
    RT   virus).";
    RL   J. Virol. 80:8439-8449(2006).
    CC   -!- SUBCELLULAR LOCATION: Host membrane {ECO:0000305}; Multi-pass membrane
    CC       protein {ECO:0000305}.
    DR   EMBL; DQ643392; ABF82052.1; -; Genomic_DNA.
    DR   RefSeq; YP_654594.1; NC_008187.1.
    DR   GeneID; 4156271; -.
    DR   KEGG; vg:4156271; -.
    DR   Proteomes; UP000001358; Genome.
    DR   GO; GO:0033644; C:host cell membrane; IEA:UniProtKB-SubCell.
    DR   GO; GO:0016021; C:integral component of membrane; IEA:UniProtKB-KW.
    PE   4: Predicted;
    KW   Host membrane; Membrane; Reference proteome; Transmembrane;
    KW   Transmembrane helix.
    FT   CHAIN           1..225
    FT                   /note="Transmembrane protein 022L"
    FT                   /id="PRO_0000377944"
    FT   TRANSMEM        2..22
    FT                   /note="Helical"
    FT                   /evidence="ECO:0000255"
    SQ   SEQUENCE   225 AA;  25107 MW;  3BD60B1CA8C7D7F5 CRC64;
         MSFVHKLPTF YTAGVGAIIG GLSLRFNGAK FLSDWYINKY NDSVPAWSLQ TCHWAGIALY
         CVGWVTLASV IYLKHRDNSI LKGSILSCIV ISAVWSILEY NQDMFVSNPK LPLISCAMLV
         SSLAALVALK YHIKDIFTIL GAAIIIILAE YVVLPYQRQY NIVDGIGLPL LLLGFFILYQ
         VFSVPNPSTP TGVMVPKPED EWDIEMAPLN HRDRQVPESE LENVK
    //

    Parameters
    ----------
    ontology: Ontology object
        ...
    swissprot_db: Path
        Path to Swissprot file

    Returns
    -------
    proteins: dict with keys of Swissprot accessions and values of Protein objects
    go_terms_not_found: list of GO terms (should be empty)
    accessions: dict with keys of Swissprot accessions and secondary accessions
                and values of Swissprot accessions
    """

    def parse(handle):
        """parse
        Inner function for parse_swissprot()
        """
        num_annotations = 0

        for rec in tqdm(
            SwissProt.parse(handle),
            desc=f"Reading SwissProt records from '{handle.name}'",
        ):
            primary_accession = rec.accessions[0]
            # Map secondary accessions to primary accession, for example:
            # AC   P0A9Q6; P08193; P76937; P78251;
            for acc in rec.accessions:
                accessions[acc] = primary_accession
            protein = Protein(primary_accession, str(rec.sequence))
            for ref in rec.cross_references:
                # Tuple of length 3, 4, or 5:
                # ('EMBL', 'JHAC01000017', 'EYB68740.1', '-', 'Genomic_DNA')
                # ('GO', 'GO:0005886', 'C:plasma membrane', 'IEA:UniProtKB-KW')
                if ref[0] == "GO":
                    go_term_id = ref[1]
                    evidence_code = ref[3].split(":")[0]
                    primary_go_term = ontology.get_primary_term(go_term_id)
                    if not primary_go_term:
                        logging.error(
                            f"GO Term {go_term_id} found in SwissProt but not "
                            "found in ontology"
                        )
                        go_terms_not_found.add(go_term_id)
                    else:
                        annot = Annotation(
                            protein.id,
                            evidence_code,
                            primary_go_term,
                        )
                        protein.add_annotation(annot)
                        num_annotations += 1

            # The protein should not already exist
            assert proteins.get(protein.id) is None
            proteins[protein.id] = protein

        logging.info(
            f"Found {len(proteins.keys())} total proteins in '{handle.name}'"
        )
        logging.info(
            f"Found {num_annotations} total annotations in '{handle.name}'"
        )

    proteins = {}
    accessions = {}
    go_terms_not_found: set[str] = set()

    if swissprot_db.suffix == ".gz":
        with gzip.open(swissprot_db, "rt") as f:
            parse(f)
    else:
        with open(swissprot_db, "r") as f:
            parse(f)

    return proteins, go_terms_not_found, accessions

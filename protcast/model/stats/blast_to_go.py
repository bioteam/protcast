import json
import logging
import os
import requests
from typeguard import typechecked
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio import Entrez

@typechecked
class BlastToGo:
    """BlastToGo
    This clast takes a protein sequence and performs a protein Blast search
    against a NCBI database, returning proteins with at least min_identity percent similarity.

    Attributes
    ----------
    protein_seq (str): 
        Path to the protein FASTA file.
    database (str): 
        Name of the NCBI protein database (e.g., "nr").
    e_value (float): 
        E-value threshold for significant alignments.
    min_identity (float): 
        Minimum percent sequence identity for hits.

    Methods
    -------
    init:
        Initialize
    blast_to_go:
        ...
    web_blast:
        ...
    get_go_from_uniprot:
        ...
    """

    def __init__(
        self,
        database: str = "nr",
        program: str = "blastp",
        e_value: float = 0.001,
        min_identity: float = 95.0,
        alignments: int = 100,
        verbose: bool = False
    ) -> None:
        """__init__

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.e_value = e_value
        self.min_identity = min_identity
        self.database = database
        self.program = program
        self.alignments = alignments
        self.verbose = verbose
        self.uniprot_url = "https://rest.uniprot.org/uniprotkb/search"

    def blast_to_go(self, seq: str) -> list[str]:
        """blast_to_go

        Parameters
        ----------

        Returns
        -------
        List of protein ids
        """
        pids = self.web_blast(seq)
        go_ids = self.get_go_from_uniprot(pids)
        return go_ids

    @typechecked
    def web_blast(self, seq: str) -> list[str]:
        """web_blast

        Parameters
        ----------

        Returns
        -------
        List of protein ids
        """
        Entrez.email = "briano@bioteam.net"
        Entrez.NCBI_API_KEY = os.getenv('NCBI_API_KEY')

        pids = list()
        # Run BLAST search using NCBIWWW
        blast_results = NCBIWWW.qblast(
            self.program,
            self.database,
            seq,
            expect=self.e_value,
            alignments=self.alignments,
        )
        blast_record = NCBIXML.read(blast_results)

        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                percent_identity = float(hsp.identities / len(hsp.query) * 100)
                # Skip 100% identity, could be the same protein
                if percent_identity < 100 and percent_identity >= self.min_identity:
                    # ref|WP_021461111.1|
                    pid = alignment.hit_id.split("|")[1]
                    pids.append(pid)
                    if self.verbose:
                        logging.debug(f"alignment.hit_id: {pid}")
        if len(pids) > 0:
            return pids

    @typechecked
    def get_go_from_uniprot(self, pids: list[str]) -> list[str]:
        """get_go_from_uniprot

        https://rest.uniprot.org/uniprotkb/search?query=WP_021461111

        Parameters
        ----------

        Returns
        -------
        List of GO ids
        """
        for pid in pids:
            response = requests.get(self.uniprot_url, params={"query": pid})
            result = json.loads(response.text)["results"]
            # UniProt may not have the id provided by NCBI Blast
            if len(result) > 0:
                """
                >>> result[0]["uniProtKBCrossReferences"][4]
                {'database': 'GO', 'id': 'GO:0005737', 'properties':
                [{'key': 'GoTerm', 'value': 'C:cytoplasm'},
                {'key': 'GoEvidenceType', 'value': 'IEA:UniProtKB-SubCell'}]}
                """
                go_ids = [
                    x["id"]
                    for x in result[0]["uniProtKBCrossReferences"]
                    if x["database"] == "GO"
                ]
                if self.verbose:
                    logging.debug(
                        f"NCBI BLAST id: {pid} UniProt primary accession: {result[0]['primaryAccession']} GO terms: {go_ids}"
                    )
                return go_ids
            else:
                if self.verbose:
                    logging.debug(f"No match in UniProt for NCBI Blast id {pid}")

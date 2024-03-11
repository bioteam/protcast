from typeguard import typechecked
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML


@typechecked
class BlastToGo:
    """BlastToGo
    This function takes a protein sequence, database name,
    e-value threshold, and minimum identity as input and performs a protein blast search
    against a NCBI database, returning proteins with at least min_identity percent similarity.

    Attributes
    ----------
    protein_seq (str): Path to the protein FASTA file.
    database (str): Name of the NCBI protein database (e.g., "nr").
    e_value (float): E-value threshold for significant alignments.
    min_identity (int): Minimum percent sequence identity for hits.

    Methods
    -------
    init:
        Initialize
    web_blast:
        ...
    """

    def __init__(
        self,
        e_value: float = 0.001,
        program: str = "blastp",
        min_identity: int = 95,
        database: str = "nr",
    ) -> None:
        self.e_value = e_value
        self.min_identity = min_identity
        self.database = database
        self.program = program

    @typechecked
    def web_blast(self, seq) -> str:
        """web_blast

        Returns:
        protein sequence with significant alignments.
        """
        # Run BLAST search using NCBIWWW
        blast_results = NCBIWWW.qblast(
            query=seq, program=self.program, database=self.database, expect=self.e_value
        )
        blast_record = NCBIXML.read(blast_results)

        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                # Skip 100% identity, could be same protein and species
                if hsp.identities / len(hsp.query) * 100 >= self.min_identity:
                    return hsp.id

from typeguard import typechecked
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from protcast.utils.uniprot import UniProt  # noqa: F401


@typechecked
class BlastToGo:
    """BlastToGo
    This class takes a protein sequence and performs a Blast search
    using the NCBI Blast API, returning proteins with at least min_identity
    percent similarity to the query. Those ids are used to query the UniProt
    API to get records which contain GO term ids. Example:

    from blast_to_go import BlastToGo
    seq = "MADTFKEIDAQNAWQLVQERQAFLVDVRDIQRFAYSHPQAAFHLTNQSYGEFCQRCDFEDPIVV"
    app = BlastToGo()
    go_ids = app.blast_to_go(seq)

    Attributes
    ----------
    protein_seq: str
        Path to the protein FASTA file
    database: str
        Name of the NCBI protein database (e.g., "nr")
    e_value: float
        E-value threshold for significant alignments
    min_identity: float
        Minimum percent sequence identity for hits
    num_alignments: int
        Number of alignments to retrieve
    verbose: bool
        Verbosity

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
        num_alignments: int = 100,
        verbose: bool = False,
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
        self.num_alignments = num_alignments
        self.verbose = verbose

    def blast_to_go(self, seq: str) -> list[str]:
        """blast_to_go
        Wrapper method for Blast API and Uniprot API queries

        Parameters
        ----------
        seq: str
            Protein sequence

        Returns
        -------
        List of GO ids
        """
        pids = self.web_blast(seq)
        go_ids = UniProt(verbose=self.verbose).get_go_from_uniprot(pids)
        return go_ids

    @typechecked
    def web_blast(self, seq: str) -> list[str]:
        """web_blast

        Parameters
        ----------
        seq: str
            Protein sequence

        Returns
        -------
        List of protein ids
        """
        pids = list()
        if self.verbose:
            print("Run BLAST search using NCBIWWW")
        blast_results = NCBIWWW.qblast(
            self.program,
            self.database,
            seq,
            expect=self.e_value,
            alignments=self.num_alignments,
        )
        blast_record = NCBIXML.read(blast_results)
        if self.verbose:
            print("Parsing BLAST search results")
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                percent_identity = float(hsp.identities / len(hsp.query) * 100)
                # Skip 100% identity, could be the same protein
                if (
                    percent_identity < 100
                    and percent_identity >= self.min_identity
                ):
                    # ref|WP_021461111.1|
                    pid = alignment.hit_id.split("|")[1]
                    pids.append(pid)
                    if self.verbose:
                        print(f"alignment.hit_id: {pid}")
        return pids

import json
import requests
from typeguard import typechecked


@typechecked
class UniProt:
    """UniProt
    UniProt API wrapper class

    Attributes
    ----------
    pids: list
        List of protein IDs

    Methods
    -------
    init:
        Initialize
    get_go_from_uniprot:
        Return GO ids from UniProt API
    """

    def __init__(
        self,
        verbose: bool = False,
    ) -> None:
        """__init__

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.uniprot_api = "https://rest.uniprot.org/uniprotkb/search"
        self.verbose = verbose

    @typechecked
    def get_go_from_uniprot(self, pids: list[str]) -> list[str]:
        """get_go_from_uniprot
        Return all GO term ids for a single protein using the UniProt API.
        The entire record as JSON, for example:

        https://rest.uniprot.org/uniprotkb/search?query=WP_021461111

        Parameters
        ----------
        pids: list[str]
            List of protein ids

        Returns
        -------
        List of GO ids or None
        """
        if self.verbose:
            print("Querying UniProt API")
        for pid in pids:
            response = requests.get(self.uniprot_api, params={"query": pid})
            result = json.loads(response.text)["results"]
            # If UniProt has the id provided by NCBI Blast
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
                    print(
                        f"NCBI BLAST id: {pid} UniProt primary accession: {result[0]['primaryAccession']} GO terms: {go_ids}"
                    )
                return go_ids
            else:
                if self.verbose:
                    print(f"No match in UniProt for NCBI Blast id {pid}")

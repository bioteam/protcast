import logging
import pickle
from typeguard import typechecked
from goatools.obo_parser import GODag
from protcast.preprocessing.annotated_goterm import AnnotatedGOTerm


class AnnotatedGODag:
    """AnnotatedGODag
    This is wrapper around the goatools GODag class which represents the 
    GO acyclic trees of GOTerms

    Attributes
    ----------
    annotations: list
        List of Annotations

    Methods
    -------
    init:
        Initialize
    """

    def __init__(self, input_file) -> None:
        """__init__
        Initialize

        Parameters
        ----------
        input_file: Path
            Input *obo file

        Returns
        -------
        None
        """
        # Use the goatools parser
        goatools_godag = GODag(input_file)
        self.parent = goatools_godag
        self.annotations = list()

        # Map GO ids to AnnotatedGOTerms
        self.go_terms_map = dict()
        for go_id, go_term in goatools_godag.items():
            self.go_terms_map[go_id] = AnnotatedGOTerm(go_term)

    # Forward attribute access to the parent object
    def __getattr__(self, item):
        return getattr(self.parent, item)

    @typechecked
    def get_term(self, go_id: str) -> AnnotatedGOTerm:
        """get_term
        Get AnnotatedGOterm given an id

        Parameters
        ----------
        go_id: str
            Id of GOTerm

        Returns
        -------
        AnnotatedGOTerm
        """
        return self.go_terms_map[go_id]
    
    def populate_annotations(self) -> None:
        """populate_annotations
        Populate annotations for a namespace

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logging.debug(f"Populating annotations of {self.name}")
        for node in self.nodes.values():
            for annot in node.annotations:
                self.annotations.append(annot)

    def save(self, output_file: str) -> str:
        """save
        Serialize an ontology

        Parameters
        ----------
        output_file: str
            Name of output file

        Returns
        -------
        File name
        """
        with open(output_file, "wb") as f:
            pickle.dump(self, f)
        return output_file

    @classmethod
    def load_ontology(cls, input_file: str):
        """load_ontology
        Read serialized ontology file

        Parameters
        ----------
        input_file: str
            Name of serialized ontology file

        Returns
        -------
        AnnotatedGODag
        """
        with open(input_file, "rb") as f:
            ontology = pickle.load(f)
        return ontology
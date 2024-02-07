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
    None

    Methods
    -------
    init:
        Initialize
    get_term:
        ...
    get_all_terms:
        Returns all AnnotatedGOTerms or by namespace
    get_all_annotations:
        Returns all Annotations or by namespace
    save:
        ...
    load_ontology:
        ...
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
        # Use goatools GODag for parsing and querying
        goatools_godag = GODag(input_file)

        # Map GO ids to AnnotatedGOTerms 
        self.go_terms_map = dict()
        for go_id, go_term in goatools_godag.items():
            self.go_terms_map[go_id] = AnnotatedGOTerm(go_term)

        # Map parents and children of goaltools GOTerm to AnnotatedGOTerm
        for go_id, annot_go_term in self.go_terms_map.items():
            # pickle.dump() recursion error if both loops are executed
            # and the object is stored rather than its id
            for parent in goatools_godag[go_id].parents:
                annot_go_term.parents.append(parent.id)
            for child in goatools_godag[go_id].children:
                annot_go_term.children.append(child.id)

        goatools_godag = None

    @typechecked
    def get_term(self, go_id: str) -> AnnotatedGOTerm | None:
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
        return self.go_terms_map.get(go_id)

    @typechecked
    def get_all_terms(self, namespace=None) -> list[AnnotatedGOTerm]:
        """get_all_terms
        Get all AnnotatedGOterms

        Parameters
        ----------
        None or namespace

        Returns
        -------
        List of AnnotatedGOTerms
        """
        if namespace:
            return [x for x in self.go_terms_map.values() if x.namespace == namespace]
        else:
            return self.go_terms_map.values()

    @typechecked
    def get_all_annotations(self, namespace=None) -> list:
        """get_all_annotations
        Get all Annotations

        Parameters
        ----------
        None or namespace

        Returns
        -------
        List of Annotations
        """
        annots = list()
        if namespace:
            for term in self.go_terms_map.values():
                if term.namespace == namespace:
                    annots.extend(term.annotations)
        else:
            for term in self.go_terms_map.values():
                annots.extend(term.annotations)
        return annots

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
    def load_godag(cls, input_file: str):
        """load_godag
        Read serialized AnnotateeGODag

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

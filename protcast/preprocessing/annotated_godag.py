from typeguard import typechecked
from goatools.obo_parser import GODag
from goatools.gosubdag.gosubdag import GoSubDag
from protcast.preprocessing.annotated_goterm import AnnotatedGOTerm

@typechecked
class AnnotatedGODag:
    """AnnotatedGODag
    This is wrapper around the goatools GODag class which represents the
    GO DAGs of GOTerms

    Attributes
    ----------
    None

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
        # Use goatools GODag for parsing and querying
        # Recursion error on pickle.dump(): self.goatools = goatools_godag
        goatools_godag = GODag(input_file)

        # Map GO ids to AnnotatedGOTerms 
        self.go_terms_map = dict()
        for go_id, go_term in goatools_godag.items():
            self.go_terms_map[go_id] = AnnotatedGOTerm(go_term)

        # Map parents and children of goaltools GOTerm to AnnotatedGOTerm
        for go_id, annot_go_term in self.go_terms_map.items():
            # pickle.dump() recursion error if both loops are executed
            # and the GO term object is stored rather than its id
            for parent in goatools_godag[go_id].parents:
                annot_go_term.parents.append(parent.id)
            for child in goatools_godag[go_id].children:
                annot_go_term.children.append(child.id)

        # Not required, no recursion error without this
        # goatools_godag = None


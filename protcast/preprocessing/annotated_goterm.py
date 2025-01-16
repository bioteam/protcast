from typeguard import typechecked
from protcast.preprocessing.annotation import Annotation


class AnnotatedGOTerm:
    """AnnotatedGOTerm
    This class is a wrapper for the goatools GOTerm class that
    adds an Annotation attribute. Only non-obsolete terms are used.

    Attributes
    ----------
    annotations: list
        List of Annotations
    level: int
        Shortest distance to root
    depth: int
        Longest distance to root
    go_id: str
        GO id
    name: str
        Description
    namespace: str
        GO namespace
    parents: list[str]
        List of GO ids
    children: list[str]
        List of GO ids

    Methodss
    -------
    init: GOTerm
        Creates AnnotatedGOTerm given a goatools GOterm
    add_annotation: Annotation
        Add Annotation to AnnotatedGOTerm
    has_annotation: Annotation
        Boolean
    get_annotation: str
        Returns Annotation given a GO id
    get_all_annotations:
        Returns all Annotations for an AnnotatedGOTerm
    get_all_sequences: dict
        Returns all sequences for all Annotations for an AnnotatedGOTerm
    """

    def __init__(self, goatools_go_term) -> None:
        """__init__
        Initialize using goatools GOTerm attributes

        Parameters
        ----------
        goatools_go_term: obj
            goatools GOTerm

        Returns
        -------
        None
        """
        # Shortest distance from root node
        self.level = goatools_go_term.level
        # Longest distance from root node
        self.depth = goatools_go_term.depth
        self.go_id = goatools_go_term.id
        self.name = goatools_go_term.name
        self.namespace = goatools_go_term.namespace
        # These will be GO ids, populated by AnnotatedGODag
        self.parents = list()
        self.children = list()
        self.annotations = list()

        goatools_go_term = None

    @typechecked
    def add_annotation(self, annot: Annotation) -> bool:
        """add_annotation
        Add Annotation to a list of Annotations if the
        AnnotatedGOterm does not have the Annotation

        Parameters
        ----------
        annot: obj
            Annotation

        Returns
        -------
        True if Annotation is added
        """
        if not self.has_annotation(annot):
            self.annotations.append(annot)
            return True
        return False

    @typechecked
    def has_annotation(self, annot: Annotation) -> bool:
        """has_annotation
        Check to see if the AnnotatedGOterm has the input Annotation

        Parameters
        ----------
        annot: ojb
            Annotation

        Returns
        -------
        Boolean
        """
        if not self.annotations:
            return False
        for term_annot in self.get_all_annotations():
            if (
                term_annot.go_id == annot.go_id
                and term_annot.evidence_code == annot.evidence_code
                and term_annot.protein_id == annot.protein_id
            ):
                return True
        return False

    def get_annotation(self, go_id: str):
        """get_annotation
        Get an Annotation from an AnnotatedGOterm given a GO id

        Parameters
        ----------
        go_id: str
            GO term id

        Returns
        -------
        Annotation or None
        """
        for annot in self.get_all_annotations():
            if annot.go_id == go_id:
                return annot

    def get_all_annotations(self):
        """get_all_annotations
        Get all Annotations for a AnnotatedGOterm

        Parameters
        ----------
        None

        Returns
        -------
        List of Annotations
        """
        if len(self.annotations) > 0:
            return self.annotations

    def get_all_sequences(self):
        """get_all_sequences
        Get all sequences for all Proteins for an AnnotatedGOterm

        Parameters
        ----------
        None

        Returns
        -------
        Dict of protein ids and protein sequences
        """
        if len(self.annotations) > 0:
            return {
                protein.id: protein.sequence
                for protein in [annot.protein for annot in self.annotations]
            }

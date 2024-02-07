from typeguard import typechecked
from protcast.preprocessing.annotation import Annotation


class AnnotatedGOTerm:
    """AnnotatedGOTerm
    This class is a wrapper for the goatools GOTerm class that
    adds an Annotation attribute.

    Attributes
    ----------
    annotations: list
        List of Annotations

    Methods
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
    """

    def __init__(self, goatools_go_term) -> None:
        """__init__
        Initialize

        Parameters
        ----------
        name: obj
            goatools GOTerm

        Returns
        -------
        None
        """
        # Use goatools GOTerm attributes
        self.goatools = goatools_go_term
        self.level = self.goatools.level
        self.go_id = self.goatools.id
        self.name = self.goatools.name
        self.namespace = self.goatools.namespace
        self.is_obsolete = self.goatools.is_obsolete
        self.depth = self.goatools.depth
        # Populated by AnnotatedGODag
        self.parents = list()
        self.children = list()

        self.annotations = list()

    @typechecked
    def add_annotation(self, annot: Annotation) -> None:
        """add_annotation
        Add Annotation to a list of Annotations if the
        AnnotatedGOterm does not have the Annotation

        Parameters
        ----------
        annot: Annotation
            ...

        Returns
        -------
        None
        """
        if not self.has_annotation(annot):
            self.annotations.append(annot)

    @typechecked
    def has_annotation(self, annot: Annotation) -> bool:
        """has_annotation
        Check to see if the AnnotatedGOterm has the input Annotation
        """
        for term_annot in self.get_all_annotations():
            if (
                term_annot.go_id == annot.go_id
                and term_annot.evidence_code == annot.evidence_code
                and term_annot.protein_id == annot.protein_id
            ):
                return True
        return False

    @typechecked
    def get_annotation(self, go_id: str) -> Annotation | None:
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
            
    @typechecked
    def get_all_annotations(self) -> list[Annotation]:
        """get_all_annotations
        Get all Annotations for a AnnotatedGOterm

        Parameters
        ----------
        None

        Returns
        -------
        List of Annotations
        """
        return self.annotations

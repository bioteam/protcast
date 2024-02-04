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
    init: obj
        goatools GOterm
    __getattr__: str
        Get any parent attribute
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
        self.parent = goatools_go_term
        self.annotations = list()

    def __getattr__(self, item):
        """__getattr__
        Forward attribute access to the parent goatools GOTerm object
        """
        return getattr(self.parent, item)

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
                term_annot.go_term_id == annot.go_term_id
                and term_annot.evidence_code == annot.evidence_code
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

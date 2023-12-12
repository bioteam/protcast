from typeguard import typechecked

from protcast.preprocessing.annotation import Annotation


@typechecked
class Protein:
    """Protein
    ....

    Attributes
    ----------
    id: str
        Protein id (Uniprot accession)
    sequence: str
        Protein sequence
    annotations: dict
        Key is GO term id, value is an Annotation

    Methods
    -------
    __init__:
        Initialize
    is_manually_annotated:
        ...
    get_electronic_non_obsolete_annotations:
        ...
    get_manual_non_obsolete_annotations:
        ...
    get_all_annotations:
        ...
    get_annotation:
        ...
    add_annotation:
        ..
    """
    def __init__(self, id: str, sequence: str) -> None:
        """__init__
        Initialize

        Parameters
        ----------
        id: str
            Protein id
        sequence: str
            Protein sequence

        Returns
        -------
        None
        """
        self.id: str = id
        self.sequence: str = sequence
        self.annotations: dict[str, Annotation] = {}

    def add_annotation(self, annot: Annotation) -> None:
        """add_annotation
        Add Annotation to a dict of Annotations for a given Protein

        Parameters
        ----------
        annot: Annotation
            ...

        Returns
        -------
        None
        """
        assert self.annotations.get(annot.go_term_id) is None
        self.annotations[annot.go_term_id] = annot

    @typechecked
    def get_annotation(self, go_term_id: str) -> Annotation | None:
        """get_annotation
        Get an Annotation given a GO term id

        Parameters
        ----------
        go_term_id: str
            GO term id

        Returns
        -------
        Annotation or None
        """
        return self.annotations.get(go_term_id)

    @typechecked
    def get_all_annotations(self) -> list[Annotation]:
        """get_all_annotations
        Get all Annotations for a Protein

        Parameters
        ----------
        None

        Returns
        -------
        List of Annotations
        """
        return list(self.annotations.values())

    def get_manual_non_obsolete_annotations(self) -> list[Annotation]:
        """get_manual_non_obsolete_annotations
        Get all manual, non-obsolete Annotations for a Protein

        Parameters
        ----------
        None

        Returns
        -------
        List of Annotations
        """
        return list(
            filter(
                lambda x: x.is_manual and not x.is_obsolete,
                self.annotations.values(),
            )
        )

    def get_electronic_non_obsolete_annotations(self) -> list[Annotation]:
        """get_electronic_non_obsolete_annotations
        Get all electronic, non-obsolete Annotations for a Protein

        Parameters
        ----------
        None

        Returns
        -------
        List of Annotations
        """
        return list(
            filter(
                lambda x: not x.is_manual and not x.is_obsolete,
                self.annotations.values(),
            )
        )

    def is_manually_annotated(self, go_term_id: str) -> bool | None:
        """is_manually_annotated
        Get is_manual for a given Annotation, or None if no Annotation
        for that GO term id

        Parameters
        ----------
        go_term_id: str
            GO term id

        Returns
        -------
        Boolean or None
        """
        annot = self.annotations.get(go_term_id)
        if not annot:
            return None
        return annot.is_manual

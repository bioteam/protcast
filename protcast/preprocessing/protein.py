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
    annotations: list[Annotation]
        list of Annotations

    Methods
    -------
    __init__:
        Initialize
    is_manually_annotated:
        Boolean
    get_electronic__annotations:
        ...
    get_manual_annotations:
        ...
    get_all_annotations:
        ...
    get_annotation:
        ...
    add_annotation:
        ...
    has_annotation
        ...
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
        self.annotations: list[Annotation] = []

    @typechecked
    def add_annotation(self, annot: Annotation) -> None:
        """add_annotation
        Add Annotation to a list of Annotations for a given Protein

        Parameters
        ----------
        annot: Annotation
            ...

        Returns
        -------
        None
        """
        self.annotations.append(annot)

    @typechecked
    def has_annotation(self, annot: Annotation) -> bool:
        """has_annotation
        Check to see if the Protein has the input Annotation
        """
        for protein_annot in self.get_all_annotations():
            if (
                protein_annot.go_term_id == annot.go_term_id
                and protein_annot.evidence_code == annot.evidence_code
            ):
                return True
        return False

    @typechecked
    def get_annotation(self, go_term_id: str) -> Annotation | None:
        """get_annotation
        Get an Annotation from a Protein given a GO term id

        Parameters
        ----------
        go_term_id: str
            GO term id

        Returns
        -------
        Annotation or None
        """
        for annot in self.get_all_annotations():
            if annot.go_term_id == go_term_id:
                return annot

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
        return self.annotations

    def get_manual_annotations(self) -> list[Annotation]:
        """get_manual_annotations
        Get all manual Annotations for a Protein

        Parameters
        ----------
        None

        Returns
        -------
        List of Annotations
        """
        return [x for x in self.get_all_annotations() if x.is_manual]

    def get_all_go_ids(self) -> list[str]:
        """get_all_go_ids
        Get all GO ids from all Annotations

        Parameters
        ----------
        None

        Returns
        -------
        List of GO ids
        """
        return [x.go_term_id for x in self.get_all_annotations()]

    def get_electronic_annotations(self) -> list[Annotation]:
        """get_electronic_annotations
        Get all electronic Annotations for a Protein (evidence code is IEA)

        Parameters
        ----------
        None

        Returns
        -------
        List of Annotations
        """
        return [
            x for x in self.get_all_annotations() if x.evidence_code == "IEA"
        ]

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

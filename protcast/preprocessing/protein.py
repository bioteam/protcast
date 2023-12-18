import logging

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
        # If an Annotation with the same pair of GO term and evidence code
        # already exists then do not add the incoming Annotation
        protein_annot = self.annotations.get(annot.go_term_id)
        if protein_annot:
            if protein_annot.evidence_code == annot.evidence_code:
                logging.debug(
                    f"Annotation already exist for Protein {self.id}: "
                    f"{annot.go_term_id} {annot.evidence_code}"
                )
                return
        self.annotations[annot.go_term_id] = annot

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

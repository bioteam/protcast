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
    accessions: list
        All accessions
    sequence: str
        Protein sequence
    annotations: list[Annotation]
        list of Annotations

    Methods
    -------
    __init__:
        Initialize
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

    def __init__(self, id: str, sequence: str, accessions: list[str]) -> None:
        """__init__
        Initialize

        Parameters
        ----------
        id: str
            Protein id
        sequence: str
            Protein sequence
        accessions: list
            One or more accessions, including the Protein id

        Returns
        -------
        None
        """
        self.id: str = id
        self.accessions: list = accessions
        self.sequence: str = sequence
        self.annotations: list[Annotation] = []

    @typechecked
    def add_annotation(self, annot: Annotation) -> None:
        """add_annotation
        Add Annotation to a list of Annotations for a given Protein

        Parameters
        ----------
        annot: obj
            Annotation

        Returns
        -------
        None
        """
        if not self.has_annotation(annot):
            self.annotations.append(annot)

    @typechecked
    def has_annotation(self, annot: Annotation) -> bool:
        """has_annotation
        Check to see if the Protein has the input Annotation
        """
        for protein_annot in self.get_all_annotations():
            if (
                protein_annot.go_id == annot.go_id
                and protein_annot.evidence_code == annot.evidence_code
            ):
                return True
        return False

    @typechecked
    def get_annotation(self, go_id: str) -> Annotation | None:
        """get_annotation
        Get an Annotation from a Protein given a GO term id

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
        return [x.go_id for x in self.get_all_annotations()]

    def get_electronic_annotations(self) -> list[Annotation]:
        """get_electronic_annotations
        Get all electronic Annotations for a Protein (evidence code IEA)

        Parameters
        ----------
        None

        Returns
        -------
        List of Annotations
        """
        return [x for x in self.get_all_annotations() if x.evidence_code == "IEA"]

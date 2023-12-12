from typeguard import typechecked


class Annotation:
    """Annotation
    Annotation objects link one protein and one GO term and store
    details on the GO annotation: is_leaf, is_obsolete, is_manual.

    Attributes
    ----------
    go_term_id : str
        ....
    protein_id : str
        ....
    is_leaf : Boolean
        ....
    is_obsolete : Boolean
        ....
    is_manual : Boolean
        Default is False

    Methods
    -------
    init:
        Initialize
    is_manual:
        Set Boolean
    """

    @typechecked
    def __init__(
        self,
        go_term_id: str,
        protein_id: str,
        is_leaf: bool,
        is_obsolete: bool,
        is_manual=False,
    ) -> None:
        """__init__
        Initialize Annotation

        Parameters
        ----------
        go_term_id: str
            ...
        protein_id: str
            ...
        is_leaf: Boolean
            ...
        is_obsolete: Boolean
            ...
        is_manual: Boolean
            Default is False

        Returns
        -------
        None
        """
        self.go_term_id = go_term_id
        self.protein_id = protein_id
        self.is_leaf = is_leaf
        self.is_obsolete = is_obsolete
        self.is_manual = is_manual

    def set_is_manual(self, is_manual: bool) -> None:
        """is_manual
        Set if the annotation is manual or not

        Parameters
        ----------
        is_manual: Boolean
            Set Boolean

        Returns
        -------
        None
        """
        self.is_manual = is_manual

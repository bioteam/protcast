from protcast.preprocessing.ontology import GOTerm
from typeguard import typechecked
import logging


class Annotation:
    """Annotation
    The Annotation object links one protein, one GO term and one evidence
    code. The Annotation also stores the Boolean for *is_manual*, *has_leaf*
    (GO term) and *has_obsolete* (GO term).

    All codes are 'manual' except for 'IEA'. Evidence Codes for Annotations
    (https://geneontology.org/docs/guide-go-evidence-codes/):

    Experimental evidence:

    Inferred from Experiment (EXP)
    Inferred from Direct Assay (IDA)
    Inferred from Physical Interaction (IPI)
    Inferred from Mutant Phenotype (IMP)
    Inferred from Genetic Interaction (IGI)
    Inferred from Expression Pattern (IEP)

    Inferred from High Throughput Experiment (HTP)
    Inferred from High Throughput Direct Assay (HDA)
    Inferred from High Throughput Mutant Phenotype (HMP)
    Inferred from High Throughput Genetic Interaction (HGI)
    Inferred from High Throughput Expression Pattern (HEP)

    Phylogenetically inferred:

    Inferred from Biological aspect of Ancestor (IBA)
    Inferred from Biological aspect of Descendant (IBD)
    Inferred from Key Residues (IKR)
    Inferred from Rapid Divergence (IRD)

    Computational analysis evidence:

    Inferred from Sequence or structural Similarity (ISS)
    Inferred from Sequence Orthology (ISO)
    Inferred from Sequence Alignment (ISA)
    Inferred from Sequence Model (ISM)
    Inferred from Genomic Context (IGC)
    Inferred from Reviewed Computational Analysis (RCA)

    Author statement evidence:

    Traceable Author Statement (TAS)
    Non-traceable Author Statement (NAS)

    Curator statement evidence:

    Inferred by Curator (IC)
    No biological Data available (ND)

    Electronic annotation evidence:

    Inferred from Electronic Annotation (IEA)

    Attributes
    ----------
    go_term_id : str
        ....
    protein_id : str
        ....
    evidence_code : str
        ....
    is_manual : bool
        ....
    has_obsolete: bool
        Has an obsolete GO term
    has_leaf: bool
        Has a leaf GO term

    Methods
    -------
    init:
        Initialize
    """

    @typechecked
    def __init__(
        self,
        protein_id: str,
        evidence_code: str,
        go_term: GOTerm,
    ) -> None:
        """__init__
        Initialize Annotation

        Parameters
        ----------
        protein_id: str
            ...
        evidence_code: str
            ...
        go_term: GOTerm
            ...

        Returns
        -------
        None
        """
        # All evidence codes except for 'IEA' are 'manual'
        MANUAL_CODES = [
            "EXP",
            "IDA",
            "IPI",
            "IMP",
            "IGI",
            "IEP",
            "HTP",
            "HDA",
            "HMP",
            "HGI",
            "HEP",
            "IBA",
            "IBD",
            "IKR",
            "IRD",
            "ISS",
            "ISO",
            "ISA",
            "ISM",
            "IGC",
            "RCA",
            "TAS",
            "NAS",
            "IC",
            "ND",
        ]

        self.go_term_id = go_term.id
        self.protein_id = protein_id
        self.evidence_code = evidence_code

        self.has_obsolete = go_term.is_obsolete

        children = go_term.get_children()
        if children:
            self.has_leaf = False
        else:
            self.has_leaf = True

        if self.evidence_code == "IEA":
            self.is_manual = False
        elif self.evidence_code in MANUAL_CODES:
            self.is_manual = True
        else:
            logging.error(
                f"Invalid evidence code: '{self.evidence_code}' (protein {self.protein_id} "
                f"GO term {self.go_term_id})"
            )
            exit(1)

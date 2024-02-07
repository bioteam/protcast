from typeguard import typechecked
import logging


class Annotation:
    """Annotation
    The Annotation object links one protein, one GO term id and one evidence
    code. The Annotation also stores the Boolean for *is_manual*.

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
    go_id: str
        ....
    protein_id: str
        ....
    evidence_code: str
        ....
    is_manual: bool
        ....

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
        go_id: str,
    ) -> None:
        """__init__
        Initialize Annotation

        Parameters
        ----------
        protein_id: str
            ...
        evidence_code: str
            ...
        go_id: str
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

        self.go_id = go_id
        self.protein_id = protein_id
        self.evidence_code = evidence_code

        if self.evidence_code == "IEA":
            self.is_manual = False
        elif self.evidence_code in MANUAL_CODES:
            self.is_manual = True
        else:
            logging.error(
                f"Invalid evidence code: '{self.evidence_code}' (protein {self.protein_id} "
                f"GO term {self.go_id})"
            )
            exit(1)

import os
import sys
import tempfile
from typeguard import typechecked
import iFeatureOmegaCLI
from ifeatpro.features import get_feature
from Bio.SeqIO import SeqRecord


class FeatureVector:
    def __init__(
        self, algorithm: str = None, feature_creator: str = None, verbose=False
    ):
        self.alg = algorithm
        self.feature_creator = feature_creator
        self.verbose = verbose
        self.algs = dict()
        self.algs["ifeatpro"] = [
            "aac",
            "apaac",
            "cksaagp",
            "cksaap",
            "ctdc",
            "ctdd",
            "ctdt",
            "ctriad",
            "dde",
            "dpc",
            "gaac",
            "gdpc",
            "geary",
            "gtpc",
            "ksctriad",
            "moran",
            "nmbroto",
            "paac",
            "qsorder",
            "socnumber",
            "tpc",
        ]
        self.algs["iFeatureOmega"] = [
            "AAC",
            "AC",
            "ACC",
            "APAAC",
            "ASDC",
            "CC",
            "CKSAAGP_type_1",
            "CKSAAGP_type_2",
            "CKSAAP_type_1",
            "CKSAAP_type_2",
            "CTDC",
            "CTDD",
            "CTDT",
            "CTriad",
            "DDE",
            "DistancePair",
            "DPC_type_1",
            "DPC_type_2",
            "GAAC",
            "GDPC_type_1",
            "GDPC_type_2",
            "Geary",
            "GTPC_type_1",
            "KSCTriad",
            "Moran",
            "NMBroto",
            "PAAC",
            "PseKRAAC_type_1",
            "PseKRAAC_type_2",
            "PseKRAAC_type_3A",
            "PseKRAAC_type_3B",
            "PseKRAAC_type_4",
            "PseKRAAC_type_5",
            "PseKRAAC_type_6A",
            "PseKRAAC_type_6B",
            "PseKRAAC_type_6C",
            "PseKRAAC_type_7",
            "PseKRAAC_type_8",
            "PseKRAAC_type_10",
            "PseKRAAC_type_11",
            "PseKRAAC_type_12",
            "PseKRAAC_type_13",
            "PseKRAAC_type_14",
            "PseKRAAC_type_15",
            "PseKRAAC_type_16",
            "QSOrder",
            "SOCNumber",
            "TPC_type_1",
            "TPC_type_2",
        ]

    @typechecked
    def get_feature_vector_names(self, feature_creator: str) -> list:
        if feature_creator == "ifeatpro":
            return self.algs["ifeatpro"]
        elif feature_creator == "iFeatureOmega":
            return self.algs["iFeatureOmega"]
        else:
            sys.exit(f"Unknown feature creator: {self.feature_creator} ")

    @typechecked
    def get_feature_vectors(self, seqs: dict) -> tuple:
        if self.alg is None:
            sys.exit("Algorithm name is required")
        if self.feature_creator == "ifeatpro":
            return self.get_ifeatpro_features(seqs)
        elif self.feature_creator == "iFeatureOmega":
            return self.get_ifeatureomega_features(seqs)
        else:
            sys.exit(f"Unknown feature creator: {self.feature_creator} ")

    def get_ifeatpro_features(self, seqs):
        """get_ifeatpro_features
        Returns a list of arrays or "feature vectors" using ifeatpro. ifeatpro
        creates a CSV file given a fasta file. The algorithms are:

        aac
        apaac
        cksaagp
        cksaap
        ctdc
        ctdd
        ctdt
        ctriad
        dde
        dpc
        gaac
        gdpc
        geary
        gtpc
        ksctriad
        moran
        nmbroto
        paac
        qsorder
        socnumber
        tpc

        Parameters
        ----------
        seqs: dict
            Key is protein id, value is protein sequence

        Returns
        -------
        List of lists of floats and list of protein ids
        """
        if self.alg not in self.algs["ifeatpro"]:
            sys.exit(f"Algorithm '{self.alg}' is not part of ifeatpro")

        ids = list()
        features = list()

        for pid, seq in seqs.items():
            tmpdir = tempfile.TemporaryDirectory()
            tmpfasta = tempfile.NamedTemporaryFile()
            # Write fasta file
            if isinstance(seq, (SeqRecord)):
                seq = str(seq.seq)
            with open(tmpfasta.name, "w") as f:
                f.write(">" + pid + "\n" + seq + "\n")
            # The try is necessary in case the input sequence is too
            # short for a given algorithm
            try:
                get_feature(tmpfasta.name, self.alg, tmpdir.name)
                # ifeatpro getfeature() creates a CSV file named by algorithm
                with open(os.path.join(tmpdir.name, self.alg + ".csv")) as f:
                    for line in f.readlines():
                        arr = line.rstrip().split(",")
                        # Skip the first column which contains the id
                        vals = [float(x) for x in arr[1:]]
                        features.append(vals)
                        ids.append(arr[0])
            except Exception as e:
                print(f"Error running {self.alg} on {pid} sequence: {e}")

        return features, ids

    def get_ifeatureomega_features(self, seqs):
        """get_ifeatureomega_features
        Returns a list of arrays or "feature vectors" using iFeatureOmega and a dict
        of protein ids and protein sequences. The 49 algorithms are:

        AAC
        AC
        ACC
        APAAC
        ASDC
        CC
        CKSAAGP_type_1
        CKSAAGP_type_2
        CKSAAP_type_1
        CKSAAP_type_2
        CTDC
        CTDD
        CTDT
        CTriad
        DDE
        DistancePair
        DPC_type_1
        DPC_type_2
        GAAC
        GDPC_type_1
        GDPC_type_2
        Geary
        GTPC_type_1
        KSCTriad
        Moran
        NMBroto
        PAAC
        PseKRAAC_type_1
        PseKRAAC_type_2
        PseKRAAC_type_3A
        PseKRAAC_type_3B
        PseKRAAC_type_4
        PseKRAAC_type_5
        PseKRAAC_type_6A
        PseKRAAC_type_6B
        PseKRAAC_type_6C
        PseKRAAC_type_7
        PseKRAAC_type_8
        PseKRAAC_type_10
        PseKRAAC_type_11
        PseKRAAC_type_12
        PseKRAAC_type_13
        PseKRAAC_type_14
        PseKRAAC_type_15
        PseKRAAC_type_16
        QSOrder
        SOCNumber
        TPC_type_1
        TPC_type_2

        Parameters
        ----------
        seqs: dict
            Key is protein id, value is protein sequence

        Returns
        -------
        List of lists of floats and a list of protein ids
        """
        if self.alg not in self.algs["iFeatureOmega"]:
            sys.exit(
                f"Algorithm '{self.alg}' is not use-able part of iFeatureOmega"
            )

        pids = list()
        features = list()

        for pid, seq in seqs.items():
            tmpfasta = tempfile.NamedTemporaryFile()
            # Create multi-fasta file from input sequences
            if isinstance(seq, (SeqRecord)):
                seq = str(seq.seq)
            with open(tmpfasta.name, "w") as f:
                f.write(">" + pid + "\n" + seq + "\n")
            # The try is necessary in case the input sequence is too
            # short for a given algorithm
            try:
                protein = iFeatureOmegaCLI.iProtein(tmpfasta.name)
                protein.get_descriptor(self.alg)
                # proteins.encodings is a dataframe, which is a dict of
                # pandas.core.series.Series objects. A pandas.core.series.Series is a one-dimensional
                # labeled array of values, similar to an array but with labeled elements. The labels
                # in these Series are the column names, like "AAC_A", "AAC_B".
                for pid, vals in protein.encodings.iterrows():
                    pids.append(pid)
                    features.append(vals.tolist())
            except Exception as e:
                print(f"Error running {self.alg} on {pid} sequence: {e}")

        if self.verbose:
            lens = [len(vec) for vec in features]
            print(f"{self.alg} feature vector length: {set(lens)}")
        return features, pids

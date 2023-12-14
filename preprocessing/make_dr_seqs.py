import os
import sys

import argparse
import logging
import subprocess
import tempfile

from Bio import SeqIO
from collections import defaultdict
from pathlib import Path
from shutil import which
from sklearn.cluster import DBSCAN

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

"""
Run the `mash` application to do pairwise protein sequence comparisons using kmers, and run 
DBSCAN from scikit-learn with the resulting distance data to identify clusters of closely related 
sequences that are removed to create a "decreased redundancy" ("dr") file. If the input file
is in Swissprot format, the removed sequences will be the ones with the fewest GO terms.

Example using a file from Uniprot:

> time python3 preprocessing-scripts/make_dr_seqs.py -c 16 -s data/uniprot_sprot_viruses.dat 
real	6m15.980s

Input file 'data/uniprot_sprot_viruses.dat' has 17039 sequences
Output file 'uniprot_sprot_viruses-dr.dat' has 10572 sequences

2137 clusters found and 6,467 sequences removed (62% of the sequences) using a EC2 r5a.4xlarge (16 cores).
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--seqfile",
    required=True,
    help="Path to input sequence file",
)
parser.add_argument(
    "--informat", default="swiss", help="Input sequence file format"
)
parser.add_argument(
    "--outformat", default="swiss", help="Output sequence file format"
)
parser.add_argument(
    "-t",
    "--threshold",
    default=0.1,
    type=float,
    help="Similarity threshold",
)
parser.add_argument(
    "-o", "--output", help="Output sequence file name"
)
parser.add_argument(
    "-c",
    "--cores",
    default="2",
    type=str,
    help="Number of cores for 'mash dist'",
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Verbose"
)
args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


def main():
    builder = MakeDRSeqs(
        args.seqfile,
        args.informat,
        args.outformat,
        args.threshold,
        args.verbose,
        args.output,
        args.cores,
    )
    builder.make_dr()


class MakeDRSeqs:
    def __init__(
        self,
        seqfile,
        informat,
        outformat,
        threshold,
        verbose,
        output,
        cores,
    ) -> None:
        self.seqfile = seqfile
        self.informat = informat
        self.outformat = outformat
        self.threshold = threshold
        self.verbose = verbose
        self.output = output
        self.cores = cores

    def make_dr(self):
        mash_out = self.run_mash()
        mash_dict = self.convert_mash_out_to_dict(mash_out)
        mat, ids = self.make_dist_matrix(mash_dict)
        clusters = self.find_dbscan_clusters(mat, ids)
        similar_seqs = self.get_similar_seqs(clusters)
        self.write_seqs(similar_seqs)

    def run_mash(self):
        """
        Tab-delimited data from `mash dist`:

        Q91G40	O55709	1	1	0/269
        Q6GZQ9	O55709	1	1	0/397
        Q6GZX4	Q6GZX4	0	0	248/248
        Q6GZX5	Q6GZX4	0	0	248/248
        Q6GZX6	Q6GZX4	0.00135228	0	245/251
        Q6GZX7	Q6GZX4	0.00598258	0	235/261

        0 is no distance (all kmers in common), 1 is no kmers in common.
        """
        self.check_for_mash()
        # Create fasta format file if input file is not fasta format
        if self.informat != "fasta":
            tmpfasta = tempfile.NamedTemporaryFile()
            if self.verbose:
                print(f"Creating fasta version of '{self.seqfile}'")
            SeqIO.convert(
                self.seqfile, self.informat, tmpfasta.name, "fasta"
            )
        tmph = tempfile.NamedTemporaryFile(delete=False)
        tmpout = open(tmph.name, "w")
        # The same file acts as both query and reference
        cmd = [
            "mash",
            "dist",
            "-i",
            "-a",
            "-p",
            self.cores,
            "-d",
            str(self.threshold),
            tmpfasta.name,
            tmpfasta.name,
        ]
        try:
            if self.verbose:
                print(f"Running 'mash dist': {cmd}")
            proc = subprocess.run(cmd, stdout=tmpout)
        except subprocess.CalledProcessError as exception:
            print(f"Error: {exception}")
            sys.exit(f"Error running 'mash dist' on {tmpfasta.name}")
        tmpout.close()
        if self.verbose:
            print(f"Completed 'mash dist', output is: {tmph.name}")
        return tmph.name

    def convert_mash_out_to_dict(self, mash_out):
        """Create a dict of dicts for the distances"""
        mash_sorted = defaultdict(dict)
        with open(mash_out, "r") as f:
            # mash_sorted = { i[0]:{i[1]:float(i[2])} for i in [l.split("\t") for l in f] }
            for l in f:
                arr = l.split("\t")
                mash_sorted[arr[0]][arr[1]] = float(arr[2])
        return mash_sorted

    def make_dist_matrix(self, mash_dict):
        """
        Example of 4 points in 2 clusters (a,b and c,d) as a square distance matrix:

           a    b    c    d
        a [0,   0.1, 1,   1],
        b [0.1, 0,   1,   1],
        c [1,   1,   0,   0.1],
        d [1,   1 ,  0.1, 0]
        """
        if self.verbose:
            print("Making distance matrix")
        # Make a dict with the position of each sequence in the matrix:
        # {CATH_HUMAN':0, CYS1_DICDI':1 ....}
        ids = {
            seqid: count
            for count, seqid in enumerate(sorted(mash_dict.keys()))
        }
        # Create a square matrix filled with 1's since most values are likely 1
        mat = [
            [1 for col in range(len(ids))] for row in range(len(ids))
        ]
        # Insert non-1 distances into the prepopulated matrix
        for count, seq1 in enumerate(sorted(mash_dict.keys())):
            for seq2 in mash_dict[seq1].keys():
                mat[count][ids[seq2]] = mash_dict[seq1][seq2]
        if self.verbose:
            print("Completed distance matrix")
        return mat, ids

    def find_dbscan_clusters(self, mat, ids):
        """
        Find clusters using DBSCAN and mash distances.
        Example using the square distance matrix above:

        >>> from sklearn.cluster import DBSCAN
        >>> clust = DBSCAN(eps=0.1,min_samples=2,metric='precomputed')
        >>> m = [[0,0.1,1,1],[0.1,0,1,1],[1,1,0,0.1],[1,1,0.1,0]]
        >>> clust.fit_predict(m)
        array([0, 0, 1, 1])
        """
        clust = DBSCAN(eps=0.1, min_samples=2, metric="precomputed")
        if self.verbose:
            print(
                f"Running DBSCAN on matrix with {len(mat[0])} sequences"
            )
        predictions = clust.fit_predict(mat)
        if self.verbose:
            print("Completed DBSCAN")
        clusters = defaultdict(list)
        id_list = list(ids)
        for count, i in enumerate(predictions):
            # Ignore -1 scores, not in any cluster
            if str(i) != "-1":
                clusters[i].append(id_list[count])
        if len(clusters) == 0:
            if self.verbose:
                print("No clusters found")
            sys.exit(1)
        if self.verbose:
            print(f"Reading '{self.seqfile}' with SeqIO")
        self.seqs = SeqIO.to_dict(
            SeqIO.parse(self.seqfile, self.informat)
        )
        if self.verbose:
            for k in clusters.keys():
                print(
                    f"Cluster {k} ({len(clusters[k])}): {clusters[k]}"
                )
                for seqid in clusters[k]:
                    print(self.seqs[seqid].description)
        return clusters

    def get_similar_seqs(self, clusters):
        """Find the sequences in a cluster with the least GO terms"""
        similar_seqs = list()
        for cluster in clusters.values():
            num_of_terms = [
                self.get_num_terms(acc) for acc in cluster
            ]
            # Index of highest number
            max_index = num_of_terms.index(max(num_of_terms))
            # Remove sequence with most terms
            cluster.pop(max_index)
            similar_seqs.extend(cluster)
        return similar_seqs

    def write_seqs(self, similar_seqs):
        # Have to use "index" since BioPython cannot write Swissprot format
        seq_dict = SeqIO.index(self.seqfile, self.informat)
        if self.verbose:
            print(
                "Input file '{self.seqfile}' has {len(seq_dict.keys())} sequences"
            )
        # For example, input is "data/viruses.dat", output is "viruses-dr.dat"
        self.output = (
            os.path.basename(self.seqfile).split(".")[0] + "-dr.dat"
        )
        with open(self.output, "w") as out:
            for seqid in seq_dict:
                if seqid not in similar_seqs:
                    out.write(seq_dict.get_raw(seqid).decode())
        if self.verbose:
            print(
                "Output file '{0}' has {1} sequences".format(
                    self.output,
                    (len(seq_dict.keys()) - len(similar_seqs)),
                )
            )

    def get_num_terms(self, seqid):
        """
        >seq.dbxrefs
        ['EMBL:AY548484', 'RefSeq:YP_031579.1', 'SwissPalm:Q6GZX4', 'GeneID:2947773', 'KEGG:vg:2947773',
        'Proteomes:UP000008770', 'GO:GO:0046782', 'InterPro:IPR007031', 'Pfam:PF04947']
        """
        seq = self.seqs[seqid]
        return len([t for t in seq.dbxrefs if t.startswith("GO:")])

    def check_for_mash(self):
        """Check whether `mash` is in PATH and is executable."""
        if which("mash") is None:
            sys.exit("'mash' is not installed or not in PATH")

    def split_results(self, mash_out):
        """Return mash results as list of lists"""
        if self.verbose:
            print("Splitting mash results")
        return [e.split("\t") for e in mash_out.split("\n") if e]

    def has_experimental_annotation(goterm, ontology):
        """Not implemented in this code"""
        return


if __name__ == "__main__":
    main()

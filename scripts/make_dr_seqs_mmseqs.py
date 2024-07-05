import os
import shutil
import sys
import time
import argparse
import logging
import subprocess
import tempfile

from io import StringIO
from Bio import SeqIO
from collections import defaultdict
from shutil import which

"""
Run the `mmseqs` application to find clusters of related sequences based on minimum sequence identity. 
All the sequences in the cluster but 1 are removed to create a "decreased redundancy" 
("dr") file. If the input file is in Swissprot format, the removed sequences will be the ones with 
the fewest GO terms.

Example using a file from Uniprot with 571609 sequences:

> time python3 scripts/make_dr_seqs_mmseqs.py -s /data/UniProt/2024-06-17/uniprot_sprot.dat
...
Reading 'uniprot_sprot_cluster.tsv'
Finding sequences to remove from 243344 clusters
Input file '/data/UniProt/2024-06-17/uniprot_sprot.dat' has 571609 sequences
Output file '/data/UniProt/2024-06-17/uniprot_sprot-dr-0.75.dat' has 243344 sequences
...
1:05:21 elapsed 
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--seqfile",
    required=True,
    help="Path to input sequence file",
)
parser.add_argument("--informat", default="swiss", help="Input sequence file format")
parser.add_argument("--outformat", default="swiss", help="Output sequence file format")
parser.add_argument(
    "-m",
    "--min-seq-id",
    default=0.75,
    type=float,
    help="Minimum sequenece identity",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
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
        args.min_seq_id,
        args.verbose,
    )
    builder.make_dr()


class MakeDRSeqs:
    def __init__(
        self,
        seqfile,
        informat,
        outformat,
        min_seq_id,
        verbose,
    ) -> None:
        self.seqfile = seqfile
        self.informat = informat
        self.outformat = outformat
        self.min_seq_id = min_seq_id
        self.verbose = verbose
        self.mmseqs_input = self.seqfile
        self.timestamp = str(time.time())
        self.output_name = os.path.basename(self.seqfile.split(".")[0])
        self.mmseqs_tsv_file = self.output_name + "_cluster.tsv"
        self.output_dir = os.path.dirname(self.seqfile)

    def make_dr(self):
        self.run_mmseqs()
        clusters = self.find_clusters()
        if self.informat == "swiss":
            self.seqs = SeqIO.to_dict(SeqIO.parse(self.seqfile, self.informat))
        seqs_to_remove = self.get_seqs_to_remove(clusters)
        self.write_seqs(seqs_to_remove)
        self.clean_up()

    def run_mmseqs(self):
        """
        Get tab-delimited data from `mmseqs easy-cluster`:
        """
        self.check_for_mmseqs()
        # Create fasta format file if input file is not fasta format
        if self.informat != "fasta":
            tmpfasta = tempfile.NamedTemporaryFile()
            if self.verbose:
                print(f"Creating fasta version of '{self.seqfile}'")
            SeqIO.convert(self.seqfile, self.informat, tmpfasta.name, "fasta")
            self.mmseqs_input = tmpfasta.name

        cmd = [
            "mmseqs",
            "easy-cluster",
            "--min-seq-id",
            str(self.min_seq_id),
            self.mmseqs_input,
            self.output_name,
            self.timestamp,
        ]
        try:
            if self.verbose:
                print(f"Running 'mmseq easy-cluster': {cmd}")
            subprocess.run(cmd)
        except subprocess.CalledProcessError as exception:
            print(f"Error: {exception}")
            sys.exit(f"Error running 'mmseq easy-cluster' on {tmpfasta.name}")
        if self.verbose:
            print("Completed 'mmseq easy-cluster'")

    def find_clusters(self):
        """Create a dict with the clusters"""
        clusters = defaultdict(list)
        if self.verbose:
            print(f"Reading '{self.mmseqs_tsv_file}'")
        with open(self.mmseqs_tsv_file, "r") as f:
            for line in f:
                arr = line.strip().split("\t")
                clusters[arr[0]].append(arr[1])
        return clusters

    def get_seqs_to_remove(self, clusters):
        """Find the sequences in a cluster with the fewest GO terms"""
        seqs_to_remove = list()
        if self.verbose:
            print(f"Finding sequences to remove from {len(clusters.values())} clusters")
        for cluster in clusters.values():
            if len(cluster) == 1:
                continue
            if self.informat == "swiss":
                num_of_terms = [self.get_num_terms(acc) for acc in cluster]
                # Index of highest number
                max_index = num_of_terms.index(max(num_of_terms))
                # Remove sequence with most terms
                if self.verbose:
                    print(f"IDs\t{cluster}\tnum_of_terms\t{num_of_terms}\tmax_index\t{max_index}")
                cluster.pop(max_index)
            else:
                # Arbitrary choice
                cluster.pop()
            seqs_to_remove.extend(cluster)
        return seqs_to_remove

    def write_seqs(self, similar_seqs):
        format_map = {"swiss": "dat", "fasta": "fa"}
        # Have to use index() since BioPython cannot write Swissprot format
        seq_dict = SeqIO.index(self.seqfile, self.informat)
        if self.verbose:
            print(f"Input file '{self.seqfile}' has {len(seq_dict.keys())} sequences")
        # For example, input is "viruses.dat", output is "viruses-dr-0.1.dat"
        self.output = (
            self.output_dir
            + "/"
            + self.output_name
            + "-dr-"
            + str(self.min_seq_id)
            + "."
            + format_map[self.outformat]
        )
        with open(self.output, "w") as out:
            for seqid in seq_dict:
                seqstr = seq_dict.get_raw(seqid).decode()
                if seqid not in similar_seqs:
                    if self.outformat != "swiss":
                        seqstr = self.swiss_to_format(seqstr)
                    out.write(seqstr)

        if self.verbose:
            print(
                f"Output file '{self.output}' has {(len(seq_dict.keys()) - len(similar_seqs))} sequences"
            )

    def clean_up(self):
        """Remove mmseqs output files"""
        shutil.rmtree(self.output_dir + self.timestamp)
        os.remove(self.output_dir + self.mmseqs_tsv_file)
        os.remove(self.output_dir + self.output_name + "_rep_seq.fasta")
        os.remove(self.output_dir + self.output_name + "_all_seqs.fasta")

    def swiss_to_format(self, str):
        """Convert between sequence formats in memory"""
        # Create StringIO objects to act as an in-memory files
        in_memory_in = StringIO(str)
        in_memory_out = StringIO()
        # Parse the string as a sequence record
        record = SeqIO.read(in_memory_in, self.informat)
        # Write the record to the in-memory file-like object
        SeqIO.write(record, in_memory_out, self.outformat)
        # Get the converted sequence data as a string
        return in_memory_out.getvalue()

    def get_num_terms(self, seqid):
        """
        >>> seq.dbxrefs
        ['EMBL:AY548484', 'RefSeq:YP_031579.1', 'SwissPalm:Q6GZX4', 'GeneID:2947773', 'KEGG:vg:2947773',
        'Proteomes:UP000008770', 'GO:GO:0046782', 'InterPro:IPR007031', 'Pfam:PF04947']
        """
        seq = self.seqs[seqid]
        return len([t for t in seq.dbxrefs if t.startswith("GO:")])

    def check_for_mmseqs(self):
        """Check whether `mmseqs` is in PATH and is executable"""
        if which("mmseqs") is None:
            sys.exit("'mmseqs' is not installed or not in PATH")

    def has_experimental_annotation(goterm, ontology):
        """Not implemented in this code"""
        return


if __name__ == "__main__":
    main()

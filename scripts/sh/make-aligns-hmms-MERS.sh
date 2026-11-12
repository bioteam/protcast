#! /bin/bash
# This file is make-aligns-hmms.sh

#SBATCH --job-name MERS-make-aligns-hmms     
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o make-aligns-hmms-MERS.out
#SBATCH -e make-aligns-hmms-MERS.err

cd $SCRATCH
#download_gb_by_taxid.py -strain MERS
#feature_to_gene_and_protein.py -strain MERS MERS/*.gb
seqs_to_aligns_and_hmms.py -verbose -strain -aligner mafft MERS MERS/ORF1*fa

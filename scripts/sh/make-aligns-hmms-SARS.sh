#! /bin/bash
# This file is make-aligns-hmms.sh

#SBATCH --job-name SARS-make-aligns-hmms
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o make-aligns-hmms-SARS.out
#SBATCH -e make-aligns-hmms-SARS.err

cd $SCRATCH
#download_gb_by_taxid.py -strain SARS
#feature_to_gene_and_protein.py -strain SARS SARS/*.gb
seqs_to_aligns_and_hmms.py -verbose -strain SARS SARS/ORF1*fa

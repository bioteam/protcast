#! /bin/bash
# This file is make-aligns-hmms.sh

#SBATCH --job-name COV2-make-aligns-hmms     
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o make-aligns-hmms-COV2.out
#SBATCH -e make-aligns-hmms-COV2.err

cd $SCRATCH
seqs_to_aligns_and_hmms.py -verbose -aligner mafft COV2/ORF1*fa

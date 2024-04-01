#!/bin/bash
# Run BinaryClassifier
#SBATCH --job-name run_ifeatpro
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_ifeatpro.out
#SBATCH -e run_ifeatpro.err
#SBATCH -N 1
#SBATCH -n 20

for ALG in "aac apaac cksaagp cksaap ctdc ctdd ctdt ctriad de dpc gaac gdpc geary gtpc ksctriad moran nmbroto paac qsorder socnumber tpc"; do
    python3 scripts/binary_classify.py -n gpcrs -t test/data/uniprotkb_gpcrs.fasta -nt test/data/uniprotkb_non-gpcrs.fasta -a $ALG
done

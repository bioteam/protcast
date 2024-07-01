#!/bin/bash
# Run BinaryClassifier
#SBATCH --job-name run_ifeatpro
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_ifeatpro.out
#SBATCH -e run_ifeatpro.err
#SBATCH -n 20

#module load all/TensorFlow/2.15.1-Python-3.10

#algorithms=(aac apaac cksaagp cksaap ctdc ctdd ctdt ctriad dde dpc gaac gdpc geary gtpc ksctriad moran nmbroto paac qsorder socnumber tpc)

algorithms=(ctriad)

for alg in "${algorithms[@]}"; do
   start_time=$(date +%s)
   python3 scripts/binary_classify.py -n gpcrs -t test/data/uniprotkb_gpcrs.fasta -nt test/data/uniprotkb_non-gpcrs.fasta -a "$alg"
   end_time=$(date +%s)
   elapsed=$((end_time - start_time))
   echo "Elapsed clock time (${alg}): $elapsed seconds"
done

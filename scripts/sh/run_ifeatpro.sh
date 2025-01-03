#!/bin/bash
# Run BinaryClassifier
#SBATCH --job-name run_ifeatpro
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_ifeatpro.out
#SBATCH -e run_ifeatpro.err
#SBATCH -n 20

source ~/.bash_profile
cd ~/git/ProtCast

module load cuda/12.2
source activate base
conda activate pytorch

pip3 install .

ALGORITHMS=(aac apaac cksaagp cksaap ctdc ctdd ctdt ctriad dde dpc gaac gdpc geary gtpc ksctriad moran nmbroto paac qsorder socnumber tpc)
FAMILY='gpcrs'

for alg in "${ALGORITHMS[@]}"; do
   if [ -s ${FAMILY}_${alg}.tsv ]; then
      continue
   fi
   start_time=$(date +%s)
   python3 scripts/binary_classify.py -n $FAMILY -t test/data/uniprotkb_gpcrs.fasta -nt test/data/uniprotkb_non-gpcrs.fasta -a "$alg"
   end_time=$(date +%s)
   elapsed=$((end_time - start_time))
   echo "Elapsed clock time (${alg}): $elapsed seconds"
done

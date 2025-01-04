#!/bin/bash
# Run BinaryClassifier on multiple sequence sets
#SBATCH --job-name run_binary_classifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_binary_classifications.out
#SBATCH -e run_binary_classifications.err
#SBATCH -n 20

$1=LEVEL
cd ${HOME}/git/ProtCast/

source ${HOME}/.bash_profile
module load cuda/12.2
source activate base
conda activate pytorch
#pip3 install .

ALGORITHMS=(aac apaac cksaagp cksaap ctdc ctdd ctdt ctriad dde dpc gaac gdpc geary gtpc ksctriad moran nmbroto paac qsorder socnumber)

for ALG in "${ALGORITHMS[@]}"; do
   for SEQFILE in $LEVEL/GO*subgraph.fa; do
      GO_ID=$(basename $SEQFILE | cut -d '_' -f1)
      if [ -s $LEVEL/${GO_ID}_${ALG}_summary.tsv ]; then
         continue
      fi
      python3 scripts/binary_classify.py -n $GO_ID -t ${GO_ID}_subgraph.fa -nt ${GO_ID}_inv_subgraph.fa -a $ALG
      mv ${GO_ID}_${ALG}_summary.tsv ${GO_ID}_${ALG}_scores.tsv ${GO_ID}_${ALG}.keras $LEVEL/
   done
done

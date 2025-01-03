#!/bin/bash
# Run BinaryClassifier on multiple sequence sets
#SBATCH --job-name run_binary_classifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_binary_classifications.out
#SBATCH -e run_binary_classifications.err
#SBATCH -n 20

cd ~/git/ProtCast/fa

source ~/.bash_profile
module load cuda/12.2
source activate base
conda activate pytorch
pip3 install .

ALGORITHMS=(aac apaac cksaagp cksaap ctdc ctdd ctdt ctriad dde dpc gaac gdpc geary gtpc ksctriad moran nmbroto paac qsorder socnumber)

for alg in "${ALGORITHMS[@]}"; do
   for seqfile in GO*subgraph.fa; do
      go_id=$(echo $seqfile | cut -d '_' -f1)
      if [ -s ${go_id}_${alg}.tsv ]; then
         continue
      fi
      python3 ../scripts/binary_classify.py -n $go_id -t ${go_id}_subgraph.fa -nt ${go_id}_inv_subgraph.fa -a $alg
   done
done

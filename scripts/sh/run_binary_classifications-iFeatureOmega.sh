#!/bin/bash
# Run BinaryClassifier
#SBATCH --job-name iFeat_run_binary_classifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o iFeat_run_binary_classifications.out
#SBATCH -e iFeat_run_binary_classifications.err
#SBATCH -n 20

DIR=binary-classification
NAME=gpcr
cd ${HOME}/git/ProtCast/

source ${HOME}/.bash_profile
module load cuda/12.2
source activate base
conda activate pytorch

ALGORITHMS=(AAC EAAC CKSAAP_type_1 CKSAAP_type_2 DPC_type_1 DPC_type_2 DDE TPC_type_1 TPC_type_2 binary binary_6bit binary_5bit_type_1 binary_5bit_type_2 binary_3bit_type_1 binary_3bit_type_2 binary_3bit_type_3 binary_3bit_type_4 binary_3bit_type_5 binary_3bit_type_6 binary_3bit_type_7 AESNN3 GAAC EGAAC CKSAAGP_type_1 CKSAAGP_type_2 GDPC_type_1 GDPC_type_2 GTPC_type_1 GTPC_type_2 AAIndex ZScale BLOSUM62 NMBroto Moran Geary CTDC CTDT CTDD CTriad KSCTriad SOCNumber QSOrder PAAC APAAC OPF_10bit OPF_10bit_type_1 OPF_7bit_type_1 OPF_7bit_type_2 OPF_7bit_type_3 ASDC DistancePair AC CC ACC PseKRAAC_type_1 PseKRAAC_type_2 PseKRAAC_type_3A PseKRAAC_type_3B PseKRAAC_type_4 PseKRAAC_type_5 PseKRAAC_type_6A PseKRAAC_type_6B PseKRAAC_type_6C PseKRAAC_type_7 PseKRAAC_type_8 PseKRAAC_type_9 PseKRAAC_type_10 PseKRAAC_type_11 PseKRAAC_type_12 PseKRAAC_type_13 PseKRAAC_type_14 PseKRAAC_type_15 PseKRAAC_type_16 KNN)

for ALG in "${ALGORITHMS[@]}"; do
   if [ -s $DIR/${NAME}_${ALG}_summary.tsv ]; then
      continue
   fi
   python3 scripts/run_binary_classification.py -n $NAME -t test/data/uniprotkb_gpcrs.fasta -nt test/data/uniprotkb_non-gpcrs.fasta -a $ALG
   mv ${NAME}_${ALG}_summary.tsv ${NAME}_${ALG}_scores.tsv $DIR/
done

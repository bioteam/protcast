#!/bin/bash
# Run MultiClassifier on multiple sequence sets
#SBATCH --job-name run_multi_classification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_multi_classification-cpu.out
#SBATCH -e run_multi_classification-cpu.err
#SBATCH -n 56
#SBATCH -N 1

DIR=multi-classification
NAME=cpu

source ${HOME}/.bash_profile
source ${HOME}/tf282/bin/activate

cd ${HOME}/git/ProtCast/

#ALGORITHMS=(AAC CKSAAP_type_1 CKSAAP_type_2 DPC_type_1 DPC_type_2 DDE TPC_type_1 TPC_type_2 GAAC CKSAAGP_type_1 CKSAAGP_type_2 GDPC_type_1 GDPC_type_2 GTPC_type_1 NMBroto Moran Geary CTDC CTDT CTDD CTriad KSCTriad SOCNumber QSOrder PAAC APAAC ASDC DistancePair AC CC ACC PseKRAAC_type_1 PseKRAAC_type_2 PseKRAAC_type_3A PseKRAAC_type_3B PseKRAAC_type_4 PseKRAAC_type_5 PseKRAAC_type_6A PseKRAAC_type_6B PseKRAAC_type_6C PseKRAAC_type_7 PseKRAAC_type_8 PseKRAAC_type_10 PseKRAAC_type_11 PseKRAAC_type_12 PseKRAAC_type_13 PseKRAAC_type_14 PseKRAAC_type_15 PseKRAAC_type_16)
ALGORITHMS=(TPC_type_2 TPC_type_1)

for ALG in "${ALGORITHMS[@]}"; do
      if [ -s $DIR/${ALG}_${NAME}.tsv ]; then
            continue
      fi
      time python3 scripts/make_multi_class_model.py -v --use_mlflow -s test/data/random-level-4.fa -a $ALG
      GOENCODER=$(echo *.pickle)
      MODEL=$(echo *.keras)
      time python3 scripts/run_multi_class_inference.py -v --use_mlflow -s test/data/random-level-4.fa -g $GOENCODER -m $MODEL >${ALG}_${NAME}.tsv
      mv $MODEL $GOENCODER ${ALG}_${NAME}.tsv ${DIR}
done

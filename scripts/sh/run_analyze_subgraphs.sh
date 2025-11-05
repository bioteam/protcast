#!/bin/bash
# Run MultiClassifier on multiple sequence sets
#SBATCH --job-name run_multi_classification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_multi_classification-gpu.out
#SBATCH -e run_multi_classification-gpu.err
#SBATCH -N 2
#SBATCH -n 32

DIR=multi-classification
NAME=gpu
CONTAINER=~/tensorflow_2.17.0-gpu.sif

source ${HOME}/.bash_profile
# Only use modules from the container
unset PYTHONPATH
module load tacc-apptainer

cd ${HOME}/git/ProtCast/

ALGORITHMS=(AAC CKSAAP_type_1 CKSAAP_type_2 DPC_type_1 DPC_type_2 DDE TPC_type_1 TPC_type_2 TPC_type_3 GAAC CKSAAGP_type_1 CKSAAGP_type_2 GDPC_type_1 GDPC_type_2 GTPC_type_1 NMBroto Moran Geary CTDC CTDT CTDD CTriad KSCTriad SOCNumber QSOrder PAAC APAAC ASDC DistancePair AC CC ACC PseKRAAC_type_1 PseKRAAC_type_2 PseKRAAC_type_3A PseKRAAC_type_3B PseKRAAC_type_4 PseKRAAC_type_5 PseKRAAC_type_6A PseKRAAC_type_6B PseKRAAC_type_6C PseKRAAC_type_7 PseKRAAC_type_8 PseKRAAC_type_10 PseKRAAC_type_11 PseKRAAC_type_12 PseKRAAC_type_13 PseKRAAC_type_14 PseKRAAC_type_15 PseKRAAC_type_16 K1TPC)
#ALGORITHMS=(CTriad)

for ALG in "${ALGORITHMS[@]}"; do
      if [ -s $DIR/${ALG}_${NAME}.tsv ]; then
         continue
      fi
      singularity exec --nv $CONTAINER python3 test/test_multi_classifier.py -v -s test/data/random-level-4_1800.fa -a $ALG
      GOENCODER=$(echo *.pickle)
      MODEL=$(echo *.keras)
      singularity exec --nv $CONTAINER python3 scripts/run_multi_class_inference.py -v -s test/data/random-level-4_200.fa -g $GOENCODER -m $MODEL > $DIR/${ALG}_${NAME}.tsv
      mv $MODEL $GOENCODER $DIR
done

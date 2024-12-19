#!/bin/bash
# Create ProtCastDataset
#SBATCH --job-name create_protcast_dataset
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o create_protcast_dataset.out
#SBATCH -e create_protcast_dataset.err
#SBATCH -N 1
#SBATCH -n 20

DATE=2024-06-17
OUTPUT_DIR=$(date +%m-%d-%Y -d now)

cd $SCRATCH
python3 $HOME/git/ProtCast/scripts/create_protcast_dataset.py \
    -o GO/$DATE/go.obo \
    -g GO/$DATE/filtered_goa_uniprot_all_noiea.gaf \
    -t UniProt/$DATE/uniprot_trembl.fasta \
    -s UniProt/$DATE/uniprot_sprot.dat \
    -O $OUTPUT_DIR
python3 $HOME/git/ProtCast/scripts/create_dataset_stats.py \
    -i $OUTPUT_DIR/ProtCastDataset.bin

#!/bin/bash
# Create SimpleDataset
#SBATCH --job-name create_simple_dataset
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o create_simple_dataset.out
#SBATCH -e create_simple_dataset.err
#SBATCH -N 1
#SBATCH -n 20

DATE=2024-01-17
OUTPUT_DIR=$(date +%m-%d-%Y -d now)

cd $SCRATCH
python3 $HOME/git/ProtCast/utility-scripts/create_simple_dataset.py \
    -o $DATE/GO/go.obo \
    -g $DATE/GO/filtered_goa_uniprot_all_noiea.gaf \
    -t $DATE/UniProt/uniprot_trembl.fasta \
    -s $DATE/UniProt/uniprot_sprot.dat \
    -O $OUTPUT_DIR
python3 $HOME/git/ProtCast/utility-scripts/create_dataset_stats.py \
    -i $OUTPUT_DIR/SimpleDataset.bin

#!/bin/bash
# Create ProtCastDataset
#SBATCH --job-name create_protcast_dataset
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o create_protcast_dataset.out
#SBATCH -e create_protcast_dataset.err
#SBATCH -N 1
#SBATCH -n 56

# Create a ProtCastDataSet at TACC, takes about 5 hours

DATE=2025-10-10
OUTPUT_DIR=$(date +%m-%d-%Y -d now)
GO_ROOT=https://release.geneontology.org
UNIPROT_ROOT=https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete

GO=$GO_ROOT/$DATE/ontology/go.obo
GAF=$GO_ROOT/$DATE/annotations/filtered_goa_uniprot_all_noiea.gaf.gz
UNIPROT_DAT=$UNIPROT_ROOT/uniprot_sprot.dat.gz
UNIPROT_FA=$UNIPROT_ROOT/uniprot_sprot.fasta.gz
TREMBL=$UNIPROT_ROOT/uniprot_trembl.fasta.gz

cd $SCRATCH || exit
mkdir -p GO/"$DATE"
mkdir -p UniProt/"$DATE"

for URL in $GO $GAF $UNIPROT_DAT $UNIPROT_FA $TREMBL; do
    FNAME=$(basename "$URL" | sed 's/.gz//')
    if [ -f GO/"$DATE"/"$FNAME" ] || [ -f UniProt/"$DATE"/"$FNAME" ]; then
        echo Found: "$FNAME"
    else
        wget -c "$URL"
        echo Downloaded: "$URL"
        if [[ "$URL" =~ gz$ ]]; then
            gunzip "$(basename "$URL")"
            echo Finished gunzip: "$FNAME"
        fi
        if [[ "$FNAME" =~ go ]]; then
            mv "$FNAME" GO/"$DATE"
        else
            mv "$FNAME" UniProt/"$DATE"
        fi
    fi
done

# Required by mmseqs:
# module load cmake/3.29.5
# module load gcc/13.2.0

python3 $HOME/git/ProtCast/scripts/make_dr_seqs_mmseqs.py \
    -s $SCRATCH/UniProt/${DATE}/uniprot_sprot.dat \
    -v \
    --min-seq-id 0.9

python3 $HOME/git/ProtCast/scripts/create_protcast_dataset.py \
    -o GO/$DATE/go.obo \
    -g GO/$DATE/filtered_goa_uniprot_all_noiea.gaf \
    -t UniProt/$DATE/uniprot_trembl.fasta \
    -s UniProt/$DATE/uniprot_sprot-dr-0.9.dat \
    -O ProtCastDataset/$OUTPUT_DIR

python3 $HOME/git/ProtCast/scripts/create_dataset_stats.py \
    -i ProtCastDataset/$OUTPUT_DIR/ProtCastDataset.bin

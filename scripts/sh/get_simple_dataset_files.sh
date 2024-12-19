#!/bin/bash
# Download files required for a ProtCastDataset
#SBATCH --job-name get_protcast_dataset_files
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o get_protcast_dataset_files.out
#SBATCH -e get_protcast_dataset_files.err
#SBATCH -N 1
#SBATCH -n 20

DATE=2024-06-17
GO_ROOT=https://release.geneontology.org
UNIPROT_ROOT=https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete

GO=$GO_ROOT/$DATE/ontology/go.obo
GAF=$GO_ROOT/$DATE/annotations/filtered_goa_uniprot_all_noiea.gaf.gz
UNIPROT_DAT=$UNIPROT_ROOT/uniprot_sprot.dat.gz
UNIPROT_FA=$UNIPROT_ROOT/uniprot_sprot.fasta.gz
TREMBL=$UNIPROT_ROOT/uniprot_trembl.fasta.gz

cd /data || exit
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

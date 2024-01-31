#!/bin/bash
# Download files required for a SimpleDataset
#SBATCH --job-name get_simple_dataset_files
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o /scratch1/04769/bosborne/get_simple_dataset_files.out

DATE=2024-01-17
GO_ROOT=https://release.geneontology.org
UNIPROT_ROOT=https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete

GO=$GO_ROOT/$DATE/ontology/go.obo
GAF=$GO_ROOT/$DATE/annotations/filtered_goa_uniprot_all_noiea.gaf.gz
UNIPROT=$UNIPROT_ROOT/uniprot_sprot.dat.gz
TREMBL=$UNIPROT_ROOT/uniprot_trembl.fasta.gz

cd $SCRATCH
mkdir -p "$DATE/GO"
mkdir -p "$DATE/UniProt"

for URL in $GO $GAF $UNIPROT $TREMBL; do
    FNAME=$(basename "$URL" | sed 's/.gz//')
    if [ -f "$DATE"/GO/"$FNAME" ] || [ -f "$DATE"/UniProt/"$FNAME" ]; then
        echo Found: "$FNAME"
    else
        wget -c "$URL"
        echo Downloaded: "$URL"
        if [[ "$URL" =~ gz$ ]]; then
            gunzip "$(basename "$URL")"
            echo Finished gunzip: "$FNAME"
        fi
    	if [[ "$FNAME" =~ go ]]; then
            mv "$FNAME" "$DATE"/GO
    	else
            mv "$FNAME" "$DATE"/UniProt
    	fi
    fi
done

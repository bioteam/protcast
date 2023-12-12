#!/bin/bash
# Download files required for a SimpleDataset

DATE=2023-11-15
GO_ROOT=https://release.geneontology.org
UNIPROT_ROOT=https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete

GO=$GO_ROOT/$DATE/ontology/go.obo
GAF=$GO_ROOT/$DATE/annotations/filtered_goa_uniprot_all_noiea.gaf.gz
UNIPROT=$UNIPROT_ROOT/uniprot_sprot.dat.gz
TREMBL=$UNIPROT_ROOT/uniprot_trembl.fasta.gz

mkdir -p "GO/$DATE"
mkdir -p "UniProt/$DATE"

for URL in $GO $GAF $UNIPROT $TREMBL; do
    FNAME=$(basename "$URL" | sed 's/.gz//')
    if [ -f GO/$DATE/"$FNAME" || -f UniProt/$DATE/"$FNAME" ]; then
        echo Found: "$FNAME"
    else
        wget "$URL"
        if [ $URL =~ gz$ ]; then
            gunzip $(basename "$URL")
            echo Finished gunzip: "$FNAME"
        fi
    fi
    if [ $FNAME =~ go ]; then
        mv $FNAME GO/$DATE
    else
        mv $FNAME UniProt/$DATE
    fi
done

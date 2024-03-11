import argparse
import json
import requests
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML


if __name__ == "__main__":
    """test_blast_to_go.py
    Run NCBI blast, get GO terms from best, non-identical match using Uniprot API
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evalue", type=float, default=0.001)
    parser.add_argument("-p", "--program", default="blastp")
    parser.add_argument("-m", "--min_identity", type=float, default=95.0)
    parser.add_argument("-d", "--database", default="nr")
    args = parser.parse_args()

    seq = "MADTFKEIDAQNAWQLVQERQAFLVDVRDIQRFAYSHPQAAFHLTNQSYGEFCQRCDFEDPIVVICYHGNSSRNVAQFLVEQGFDEVYSVRGGFDAWCKAELPLEQGL"

    uniprot_url = "https://rest.uniprot.org/uniprotkb/search"

    blast_results = NCBIWWW.qblast(args.program, args.database, seq, expect=args.evalue)
    blast_record = NCBIXML.read(blast_results)

    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            percent_identity = float(hsp.identities / len(hsp.query) * 100)
            # Skip 100% identity, could be the same protein 
            if percent_identity < 100 and percent_identity >= args.min_identity:
                # ref|WP_021461111.1|
                pid = alignment.hit_id.split("|")[1]
                # https://rest.uniprot.org/uniprotkb/search?query=WP_021461111
                response = requests.get(uniprot_url, params={"query": pid})
                result = json.loads(response.text)["results"]   
                if len(result) > 0:
                    """
                    result[0]["uniProtKBCrossReferences"][4]
                    {'database': 'GO', 'id': 'GO:0005737', 'properties': [{'key': 'GoTerm', 'value': 'C:cytoplasm'}, 
                    {'key': 'GoEvidenceType', 'value': 'IEA:UniProtKB-SubCell'}]}
                    """
                    go_ids = [x["id"] for x in result[0]["uniProtKBCrossReferences"] if x["database"] == "GO"]
                    acc = result[0]["primaryAccession"]
                    print(f"Id: {acc} {go_ids}")
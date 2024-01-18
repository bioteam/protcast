# ProtCast
Extract protein sequences and associated Gene Ontology (GO) annotation from UniProt, TrEMBL and the Gene
Ontology Annotation database and use feature vector representations of the protein sequences to predict
Molecular Function, Cellular Component, and Biological Process of uncharacterized proteins. This code 
uses Keras and its FeatureSpace package for structured (tabular) data classification.

# Usage

## Building the SimpleDataset
A SimpleDataset combines protein sequences and GO annotations from multiple input files. A typical
SimpleDataset has ~0.5M proteins and ~3M GO annotations. The SimpleDataset is used as input to
FeatureSpace for processing and subsequent model-building by Keras.

### Data Sources
Input files can be downloaded using `utility-scripts/get_simple_dataset_files.sh`. The
largest file comes from TrEMBL and it's ~55GB in size.

#### UniProt/Swiss-Prot
A manually curated, high-quality protein database made by extensive annotation and expert review. The
latest version of the database can be found
[here](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz).

#### UniProtKB/TrEMBL 
The UniProt/TrEMBL database is used to retrieve the AA sequences of proteins
that are annotated in the UniProt-GOA database. Due to its size it is not
tracked in this repository. The latest release of the database can be found
[here](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz)
and it was used to obtain the sequences found in UniProt-GOA database.

#### Gene Ontology (GO)
The Gene Ontology databases `(.obo)` are downloaded from:

https://release.geneontology.org

#### UniProt-GOA
UniProt-GOA is a database that links the Gene Ontology database described
above with gene products (i.e. genes and any entities encoded by the gene
such as protein or functional RNAs)[2]. These links are what the Gene Ontology
Consortium (GOC) calls annotations which are associations between biomolecules 
and the GO terms. It is important to note that annotations contain an evidence code
which describes the origin of the annotation such as experimental,
computational analysis or electronic. For a more in-depth explanation and
documentation of the GO databases please visit the GOC website. The GOC
publishes two types of GOA-UniProt databases:

- Filtered: does not contain annotations for those species where a different
  Consortium group is primarily responsible for providing GO annotations and
  also excludes annotations made using automated methods
- Unfiltered: Contains annotations from a larger number of species and
  includes automation generated annotations.

### Processing
The steps to build the dataset to the model are:



## Building Model Dataset



## Create a Model



## Building the Ontology
Generating the ontology: `protcast/preprocessing/ontology.py` takes a GO
ontology (that can be downloaded from: http://current.geneontology.org/ontology/go.obo)
and creates an instance of the `Ontology` class.
In addition, the script can also serialize
and save the ontology into a file which can then be deserialized using
`load_ontology()`.

### Ontology
The GO terms are generated from parsing an `.obo` file and the following data
is obtained:

- GO term id
- GO term namespaces: biological process (BP), cellular component (CC),
  molecular funcion (MF)
- GO term parents
- GO term is obsolete. Check for the presence of the line starting with
  "is obsolete:"

## Parsing SwissProt

`preprocessing/parse_swissprot.py` takes a GO ontology file
and a Swissprot database file and converts it into an instance of a `Dataset`
class. The database was downloaded from:
[here](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz).
Due to file size it is not added to the repository.

For example:

```
python3 preprocessing/parse_swissprot.py \
  data/ontology/go.obo \
  data/uniprot/uniprot_sprot.dat
```

- Feature vector name: There are multiple feature vectors that can be
  generated from a protein sequence such as ctriad, PAAC or SPMAP.

# Classes

## GOTerm 
This class represents a GO term. Its attributes include *id*, *namespace*,
*name*, its *level* in the DAG, *parents*, *ancestors*, and *annotations*. The 
object also contains *is_obsolete* and *primary*.

## Annotation
This class represents an annotation and its attributes including *go_term_id*,
*protein_id*, and *evidence_code*. The object also specifies *is_manual* 
(evidence code) and *has_obsolete* (GO term).

## GODAG 
This class represents a typical OBO ontology, a directed acyclic graph. Its
attributes include *nodes* (GOTerms) and *name*.

## Ontology
This class reads input files and creates a GODAG for each of the GO namespaces.
Its attributes include *bp_dag*, *cc_dag*, *mf_dag*, and *terms*, a dict where
the values are GOTerms.

## Protein
This class represent a protein, including its *id*, *sequence*, and *annotations*. 

## SimpleDataset
This class integrates all the classes above. Its attributes include *proteins*, 
*accessions* (relating primary and secondary protein ids), different input file names 
(*ontology_path*, *swissprot_path*, *trembl_path*, *gaf_path*), and *output_dir*,
the location of the serialized SimpleDataset and log files.

## UML Component Diagrams
The UML diagrams were generated using the online tool: http://www.plantuml.com/

The following digram describes the sources:

![Alt text](dataset_generation.png?raw=true)

# Repo Organization

├- protcast/

├── doc/

├── preprocessing/

├── test/

├── training-scripts/

├── utility-scripts/

## `protcast``

`protcast/model`

`protcast/preprocessing`



`protcast/stats`

Scripts that can provide statistics on datasets.

## `doc`

Schema images.

## `preprocessing`
Contains the code that parses the raw data, builds the python
structures to represent the inputs for the model and converts them into a
format that can be used to feed the model for training. More specifically:

- It contains the code that parses the GO database and creates a DAG for each
of the GO categories.

For example:

```
python3 preprocessing/stats/create_dataset_stats.py \
  data/dataset/dataset.bin \
  -d data/dataset/stats/ -w
``` 

Contains the scripts to generate the input SimpleDataset.

Pre-processing converts the data from the source databases (UniprotKB/GO/GOA)
into the proper format to be fed to model for training, evaluation and
prediction. The preprocessing consists of:

  - Parsing the Gene Ontology (GO) database and creating a representation
    of it.
  - Parsing the UniprotKB databse and create a representation of the proteins
    in along with its annotations.
  - Given the two representations above, create the submodels that make up
    network. This will be determined from the category of the GO term, its
    number of associations and its level in the tree.
  - Generating the feature vectors from the protein sequences. Initially,
    CTriad will be used.
  - Split of the dataset into training/validation/test datasets.

## `test`
Simple test scripts. `test/data/` directory contains an `.obo` file and small, 
representative test files. For example:

```
cd test/
python3 test_build_ontology.py
python3 test_gaf_parser.py
python3 test_create_simple_dataset.py
```

## `training-scripts`
Scripts that can build networks.

## `utility-scripts`
Useful scripts. For example, download the 4 input files necessary to build
a SimpleDataset:

```
./get_simple_dataset_files.sh
```

<<<<<<< HEAD
# Environment

## imblearn Package
This code uses the MLSMOTE algorithm which is currently not part of the
latest release of the `imblearn` library. Thus, in order to use MLSMOTE it is
necessary to install it from a custom branch. To install follow these steps:

1. Clone the `balvisio` fork of imblearn:
```
git clone git@github.com:balvisio/imbalanced-learn.git
```

2. Checkout the `ba/MLSMOTE` branch:
```
cd imbalanced-learn/
git checkout ba/MLSMOTE
```

3. Install the library from local
```
python3 -m pip install .
```
=======
`make_dr_seqs.py`
Runs the `mash` application to do pairwise protein sequence comparisons using kmers, then runs 
DBSCAN from scikit-learn with the resulting distance data to identify clusters of closely related 
sequences that are removed to create a "decreased redundancy" (dr) file. If the input file
is in Swissprot format, the removed sequences will be the ones with the fewest GO terms.

>>>>>>> bio/README

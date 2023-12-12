# ProtCast



# Implementation

This implementation uses the Keras library.

# Repo Organization

├- protcast/

├── doc/

├── preprocessing-scripts/

├── test/

├── training-scripts/

├── utility-scripts/

├── LICENSE

└── README.md

`protcast/model`



`protcast/preprocessing`

Contains the code that parses the databases/raw data, builds the python
structures to represent the inputs for the model and converts them into a
format that can be used to feed the model for training. More specifically:

- It contains the code that parses the GO database and creates a DAG for each
of the GO categories.

For example:

```
python preprocessing-scripts/stats/create_dataset_stats.py \
  data/dataset/dataset.bin \
  -d data/dataset/stats/ -w
```

`protcast/stats`

Scripts that can provide statistics on datasets.

`doc`

Schema images.

`preprocessing-scripts`

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

`test`

Simple test scripts.

`training-scripts`

Scripts that can build  networks.

# Environment

## imblearn Package

This repository uses the MLSMOTE algorithm which is currently not part of the
latest release of the `imblearn` library. Thus, in order to use MLSMOTE it is
necessary to install it from a custom branch. Inside the conda environment
created for this project, follow the steps:

1. Clone the `balvisio` fork of imblearn:
```
git clone git@github.com:balvisio/imbalanced-learn.git
```

2. Checkout the `ba/MLSMOTE` branch:
```
git checkout ba/MLSMOTE
```

3. Install the library from local
```
pip install .
```

# Usage

## Building the Input Dataset

### Sources

### UniProtKB/Swiss-Prot Database



### UniProtKB/TrEMBL Database
The UniProt/TrEMBL database is used to retrieve the AA sequences of proteins
that are annotated in the UniProt-GOA database. Due to its size it is not
tracked in this repository. The latest release of the database can be found
[here](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz)
and it was used to obtain the sequences found in UniProt-GOA database.


### Gene Ontology (GO)
The Gene Ontology databases `(.obo)` were downloaded from:



#### UniProt-GOA

UniProt-GOA is a database that links the Gene Ontology database described
above with gene products (i.e. genes and any entities encoded by the gene
such as protein or functional RNAs)[2]. These links are what the Gene Ontology
Consortium (GOC) calls annotations which are associations between gene
products and the GO terms. The sources of these annotations are collaborating
databases. It is important to note that annotations contain an evidence code
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

Generating the ontology: `preprocessing-scripts/build_ontology.py` takes a GO
ontology (that can be downloaded from: http://current.geneontology.org/ontology/go.obo)
and creates an instance of the `Ontology` class defined in
`ontology/ontology.py`. In addition, the script can also serialize
and save the ontology into a file which can then be deserialized using
`load_ont()`.

`data/ontology` directory contains both the original `.obo` file and the
serialized ontology used to train the protein predictor.

For example:

```
python preprocessing-scripts/ontology.py \
  data/ontology/go_20210901.obo \
  data/ontology/serialized/go.bin
```

## Parsing SwissProt

`preprocessing-scripts/parse_swissprot.py` takes a serialized GO ontology file
and a Swissprot database file and converts it into an instance of a `Dataset`
class. The database was downloaded from:
https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz.
Due to file size it is not added to the repository.

For example:

```
python preprocessing-scripts/parse_swissprot.py \
  data/ontology/serialized/go.bin \
  data/uniprot/uniprot_sprot.dat
```


## Ontology

The GO terms are generated from parsing an `.obo` file and the following data
is obtained:
- GO term id
- GO term namespaces: biological process (BP), cellular component (CC),
  molecular funcion (MF)
- GO term parents
- GO term is obsolete. Check for the presence of the line starting with
  "is obsolete:"



- Feature vector name: There are multiple feature vectors that can be
  generated from a protein sequence such as ctriad, PAAC or SPMAP.
- Buckets: It is a list of integers that specifies the boundaries for the
  buckets in which the GO terms can be placed. For example if
  `buckets = [30, 100, 200, 500]` then, GO terms with less than 30 annotated
  proteins will be filtered out. Meanwhile, GO terms with 30 to 100 annotated
  proteins will belong to the same bucket when creating the submodels (dense
  networks). Idem, for 100 to 200, 200 to 500 and more than 500.

## GOTerm Class


## Annotation Class


## GODAG Class

A class to represent a typical OBO ontology and present relevant methods.

## Ontology Class

This class reads input files and creates a GODAG for each of the GO namespaces.

The following digram describes the sources:

![Alt text](dataset_generation.png?raw=true)

## UML Component Diagrams

The UML diagrams were generated using the online tool: http://www.plantuml.com/

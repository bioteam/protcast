# ProtCast



# Implementation

This implementation uses the Keras library.
Some definitions/conventions and functions were obtained from:

https://github.com/cansyl/DEEPred

# Repo Organization

├- protcast/

├── doc/

├── preprocessing-scripts/

├── test/

├── training-scripts/

├── utility-scripts/

├── LICENSE

└── README.md

`data`

Directory for input data (e.g. *obo, *dat, *gaf) databases and datasets used
for training and testing. Currently, the big data files are stored in an S3
bucket named "bioteam-protcast" using DVC to track them.

`protcast/model`

Contains the implementation of DeepRed.

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

Contains the scripts to generate the input Datasets and DeepredDatasets.

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

Scripts that can build Deepred networks.

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

# dvc

This repository makes use of `dvc`. Make sure sure you have `dvc` installed
in your machine:

Mac OS:

```
brew install dvc
```

## Building the Input Dataset

### Sources

### UniProtKB/Swiss-Prot Database
The `data/uniprotkb` contains two directories: `t0` and `t1`. Time point `t0`
is the deadline to submit the prediction models while `t1` is when the
benchmarks are collected for the evaluation of the submmitted models. For this
repository we set `t0` to be November 2021 and `t1` July 2022.
Each of these directories contains two files:
  - `uniprot_sprot-only<YYYY_MM>.tar.gz`: This is the Uniprot/Swiss-prot
  database used to generate the datasets to train and benchmark DEEPred.
  - `uniprot_sprot.dat`: This file is the `text/dat` format of the
  Swiss-Prot database. It comes from untaring and gunziping the file above.

  - For `t0`:
    -  [uniprot_sprot-only2021_04.tar.gz]
    (https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2021_04/knowledgebase/uniprot_sprot-only2021_04.tar.gz)
    `md5 = bd2b1b6b0a0027f674017fe5b41fadcb`.
    - `uniprot_sprot.dat`: `md5 = 057c566360413c774aed21db30253320`

  - For `t1`:
    - [uniprot_sprot-only2022_07.tar.gz]
    (https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2022_02/knowledgebase/uniprot_sprot-only2022_02.tar.gz)
    `md5 = c3e8dd5b138f0795aeba2b701982ac2c`.
    - `uniprot_sprot.dat`: `md5 = 0d328d4b468f616baf459fb71fda3597`

### UniProtKB/TrEMBL Database
The UniProt/TrEMBL database is used to retrieve the AA sequences of proteins
that are annotated in the UniProt-GOA database. Due to its size it is not
tracked in this repository. The latest release of the database can be found
[here](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz)
and it was used to obtain the sequences found in UniProt-GOA database.

The UniProtKB/TrEMBL release of April, 2021 which is the latest available
release before September, 2021 (`t0`) can be downloaded from
[here](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2021_04/knowledgebase/knowledgebase2021_04.tar.gz).
Note that the FASTA format is not available in the archives.

`data/trembl/uniprot_kb_goa_2021_10_26_seqs.fasta` is a filtered version of
the TrEMBL database that contains the TrEMBL sequences present in
`data/goa/t0/goa_uniprot_all_noiea.gaf`. This database can be useful when
generating a new `Dataset`. The filtered database was generated using by
`preprocessing-scripts/filter_fasta_from_goa.py`.

### Gene Ontology (GO)
The Gene Ontology databases `(.obo)` were downloaded from:

- For `t0`: [here](http://release.geneontology.org/2021-10-26/ontology/go.obo).
  The md5 sum of the file is `6757c819642e79e1406cad3ffcb6ea3d`.
- For `t1`: [here](http://release.geneontology.org/2022-07-01/ontology/go.obo).
  The md5 sum of the file is `1b557078fdb541dbed5ee3fb1f51cbed`.

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

The Uniprot-GOA databases are saved in `data/goa`. To generate our dataset
we used the filtered version of the UniProt-GOA since it was observed in the
DEEPRed paper that using a dataset containing electronic annotations
negatively affected the performance of the model. The database
file name used is goa_uniprot_all_noiea.gaf.gz. For this project we used:

- For `t0` the version from OCtober 26th, 2021 which was downloaded from [here]
(http://release.geneontology.org/2021-10-26/annotations/goa_uniprot_all_noiea.gaf.gz)

The md5 of the files are:
- `goa_uniprot_all_noiea.gaf.gz`: `6d2be8b48dd0f2505092a8a2cadabd6e`
- `goa_uniprot_all_noiea.gaf`: `8b2d3720f7b3f7ef63352cfec6b9d1a3`

- For `t1` the version from July 1st, 2022 which was downloaded from [here]
(http://release.geneontology.org/2022-07-01/annotations/goa_uniprot_all_noiea.gaf.gz).
The md5 of the files are:
- `goa_uniprot_all_noiea.gaf.gz`: `6f778bdc0b0ea0978e7d75f5e5289064`
- `goa_uniprot_all_noiea.gaf`: `8bc9d84bc9643dae178e33e994ccc2fb`

### Processing
The steps to build the dataset to the model are:

1. `preprocessing-script/protcast_dataset.py`: Takes an ontology file (`.obo`),
a SwissProt file (`.dat`), and a GOA (`.gaf` file) and creates a "generic"
dataset. It doesn't filter GO terms with low number of associated proteins
and doesn't group GO terms into buckets. For example:

```
python preprocessing-scripts/create_protcast_dataset.py \
  data/ontology/go_20210901.obo data/uniprot/uniprot_sprot.dat \
  data/goa/goa_uniprot_all_noiea.gaf \
  -o data/dataset/dataset.bin
```

2. Takes a `Dataset` (generated in step 1) and creates data files (`.mat`)
that can be used to train the model. It takes the following parameters:

- Minimum protein association threshold: Minimum number of proteins that a
  GO term needs to be associated with to be included in the model.
- Buckets: 
  - TODO

## Building Model Dataset

```
python preprocessing-scripts/create_protcast_dataset.py \
  -f data/dataset/dataset.bin \
  -o data/model-dataset/model_dataset.bin
```

## Create a Model

TODO

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

# Contributing

The `protein predictors` in BioTeam are happy to receive contributions.
In general, we follow the *fork-and-pull* Git workflow:

1. **Fork** the repo on GitHub
2. **Clone** the project to your own machine
3. **Commit** changes to your own branch
4. **Push** your work back up to your fork
5. Submit a **Pull request** so that we can review your changes


## Ontology

The GO terms are generated from parsing an `.obo` file and the following data
is obtained:
- GO term id
- GO term namespaces: biological process (BP), cellular component (CC),
  molecular funcion (MF)
- GO term parents
- GO term is obsolete. Check for the presence of the line starting with
  "is obsolete:"


## Generic Dataset vs. Deepred Dataset

This repository contains two closely related classes: a generic dataset
(`Dataset` class) and a Deepred Dataset (`DeepredDataset` class):

## Dataset Class

This class is composed by a GO Ontology, proteins and annotations. The
annotations is what links the proteins with the GO annotations. The dataset
is composed of:

- Dictionary of proteins found in Swissprot parsed from the uniprot file (`.dat` file)
- Three GO trees: BP, CC and MF parsed from the ontology file (`.obo` file)
- Proteins and GO Terms have annotations to link them. Annotations can either
  be manual or electronic.

The dataset holds the following metadata:

- Date of creation
- Path to the ontology file (`.obo`), uniprot file (`.dat`) and GOA
  file (`.goa`) along with their `md5` sums.

### Dataset Generation

The dataset was generated using three databases:
- Gene Ontology
- Uniprot
- Uniprot-GOA

## DeepredDataset Class

This class builds on top of a Dataset and has parameters specific for training
DeepRed:

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

# Future Work

- Detailed performance analysis/comparison with benchmarks.
- Use and evaluation of different input feature vectors such as SPMAP or PAAC.

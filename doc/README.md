

# Documentation

├── data/                  # 
├── protcast/               # 
├── doc/                   # 
├── preprocessing-scripts/ # 
├── test/                  # 
├── training-scripts/      # 
├── .gitattibutes           
├── .gitignore
├── LICENSE
└── README.md

# Pre-processing

Pre-processing converts the data from the source databases (UniprotKB/GO/GOA) into the 
proper format to be fed to model for training, evaluation and prediction. The preprocessing 
consists of:

  - Parsing the Gene Ontology (GO) database and creating a representation of it.
  - Parsing the UniprotKB databse and create a representation of the proteins
    in along with its annotations.
  - Given the two representations above, create the submodels that make up
    ProtCast. This will be determined from the category of the GO term, its number
    of associations and its level in the tree.
  - Generating the feature vectors from the protein sequences. Initially, CTriad?
     will be used.
  - Split of the dataset into training/validation/test datasets.

## Ontology

The GO terms are generated from parsing an `.obo` file and the following data is obtained:
- GO term id
- GO term namespaces: biological process (BP), cellular component (CC), molecular funcion (MF)
- GO term parents
- GO term is obsolete. Check for the presence of the line starting with "is obsolete:"


## Generic Dataset vs. Oracle Dataset

This repository contains two closely related classes: a generic dataset (`Dataset` class) and 
Oracle Dataset (`OracleDataset` class):
 
## Dataset Class

This class is composed by a GO Ontology, proteins and annotations. The annotations is what 
links the proteins with the GO annotations. The dataset is composed of:

- Dictionary of proteins found in Swissprot parsed from the uniprot file (`.dat` file)
- Three GO trees: BP, CC and MF parsed from the ontology file (`.obo` file)
- Proteins and GO Terms have annotations to link them. Annotations can either be manual 
  or electronic.

The dataset holds the following metadata:

- Date of creation
- Path to the ontology file (`.obo`), uniprot file (`.dat`) and GOA file (`.goa`) along 
  with their `md5` sums.

## DeepredDataset Class

This class builds on top of a Dataset and has parameters specific for training DeepRED:

- Feature vector name: There are multiple feature vectors that can be generated from a 
  protein sequence such as ctriad, PAAC or SPMAP.
- Buckets: It is a list of integers that specifies the boundaries for the buckets in which 
  the GO terms can be placed. For example if `buckets = [30, 100, 200, 500]` then, GO terms 
  with less than 30 annotated proteins will be filtered out. Meanwhile, GO terms with 30 to 
  100 annotated proteins will belong to the same bucket when creating the submodels (dense 
  networks). Idem, for 100 to 200, 200 to 500 and more than 500.

## GOTerm Class


## Annotation Class


## GODAG Class

A class to represent a typical OBO ontology and present relevant methods.

## Ontology Class

This class reads input files and creates a GODAG for each of the GO namespaces.

## Dataset Generation

The dataset was generated using three databases:
- Gene Ontology
- Uniprot
- Uniprot-GOA

The following digram describes the sources:

![Alt text](dataset_generation.png?raw=true)


## UML Component Diagrams

The UML diagrams were generated using the online tool: http://www.plantuml.com/


# Issues

- why 'lambda', should be list comprehensions, e.g. get_manual_annotations, 
  get_non_obsolete_annotations, get_manual_non_obsolete_annotations

- create_dataset.py refers to a non-existent script:

"See the script "create_model_dataset.py", '-f' flag."

- preprocessing-scripts/annotate_proteins.py does not create file output, just 
  calls annotate_proteins_from_goa(), without assert()

- train_submodel.py refers to OracleDataset, no such class


# Future Work

- Detailed performance analysis/comparison with benchmarks.
- Use and evaluation of different input feature vectors such as SPMAP or PAAC.
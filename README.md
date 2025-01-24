# ProtCast

Extract protein sequences and associated Gene Ontology (GO) annotations from UniProt, TrEMBL and
Gene Ontology Annotation files and use feature vector representations of the protein sequences to
predict the Molecular Function, Cellular Component, and Biological Process GO terms of proteins.
This code uses Keras and its FeatureSpace package for structured (tabular) data classification.

## Installation

### iFeatureOmega-CLI Package

This code uses a fork of the iFeatureOmega-CLI package to create feature vectors from protein sequence.

1.Clone the `bosborne` fork of iFeatureOmega-CLI:

```shell
git clone https://github.com/bosborne/iFeatureOmega-CLI
```

2.Install the library from the local directory:

```shell
cd iFeatureOmega-CLI
pip3 install .
```

### Install ProtCast

```shell
git clone https://github.com/bioteam/ProtCast
cd ProtCast
pip3 install .
```

## Building the ProtCastDataset

A ProtCastDataset combines protein sequences and GO annotations from multiple input files. A typical
ProtCastDataset has ~0.5M proteins and ~3M GO annotations. The ProtCastDataset is used as input to
FeatureSpace for processing and subsequent model-building by Keras.

### Data Sources

All the input files can be downloaded using `scripts/sh/get_protcast_dataset_files.sh`. The
largest file comes from TrEMBL and it's ~55GB in size.

#### UniProt/Swiss-Prot

A manually curated, high-quality protein database made by extensive annotation and expert review. The
latest version of the database can be found at
[ftp.uniprot.org](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz).

#### UniProtKB/TrEMBL

The UniProt/TrEMBL database is used to retrieve the AA sequences of proteins
that are annotated in the UniProt-GOA database. Due to its size it is not
tracked in this repository. The latest release of the database can be found at
[ftp.uniprot.org](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz)
and it was used to obtain the sequences found in UniProt-GOA database.

#### Gene Ontology (GO)

The Gene Ontology databases `(.obo)` are downloaded from [release.geneontology.org](https://release.geneontology.org).

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
ontology (that can be downloaded from: <http://current.geneontology.org/ontology/go.obo>)
and creates an instance of the `Ontology` class.
In addition, the script can also serialize
and save the ontology into a file which can then be deserialized using
`load_ontology()`.

### Ontology

GO file parsing is done by the [goatools](https://github.com/tanghaibao/goatools) package.

## Parsing SwissProt

`preprocessing/parse_swissprot.py` takes a GO ontology file and a Swissprot
database file and returns the proteins, the GO terms that are not found in
Swissprot, and the protein accessions. For example:

```shell
python3 preprocessing/parse_swissprot.py data/ontology/go.obo data/uniprot/uniprot_sprot.dat
```

## Classes

### Annotated_GOTerm

This class represents a GO term. Its attributes include *id*, *namespace*,
*name*, its *level* in the DAG, *parents*, *ancestors*, and *annotations*. The
object also contains *is_obsolete* and *primary*.

### Annotation

This class represents an annotation and its attributes including *go_id*,
*protein_id*, and *evidence_code*. The object also specifies *is_manual*
(evidence code) and *has_obsolete* (GO term).

### Annotated_GODAG

This class represents an OBO ontology plus associated Annotations. It is a directed
acyclic graph. Its attributes include *nodes* (GOTerms) and *name*.

### Protein

This class represent a protein, including its *id*, *sequence*, and *annotations*.

### ProtCastDataset

This class integrates all the classes above. Its attributes include *proteins*,
*accessions* (relating primary and secondary protein ids), different input file names
(*ontology_path*, *swissprot_path*, *trembl_path*, *gaf_path*), and *output_dir*,
the location of the serialized ProtCastDataset and log files.

### UML Component Diagrams

The UML diagrams were generated using the online tool: <http://www.plantuml.com/>

The following digram describes the sources:

![Alt text](dataset_generation.png?raw=true)

## Repo Organization

├- protcast/ The package directory.

├─ doc/

├─ test/

├─ scripts/

### `protcast`

#### `protcast/model`

#### `protcast/model/stats`

#### `protcast/preprocessing`

Contains the classes that parses the raw data, builds the python
objects to represent the inputs to the models and converts them into a
format that can be used for training. More specifically:

- It contains the code that parses the GO database and creates a DAG for each
of the GO categories.

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

##### `protcast/preprocessing/stats`

Scripts that can provide statistics on datasets.

#### `protcast/utils`

## `doc`

Schema images.

## `test`

Simple test scripts. The `test/data/` directory contains an `.obo` file and small,
representative test files.

## `scripts`

Scripts to preprocess and then build Keras models.

### `create_dataset_stats.py`

### `create_query_sequences_files.py`

### `create_protcast_dataset.py`

For example:

```shell
python3 scripts/create_protcast_dataset.py \
data/dataset/dataset.bin \
-d data/dataset/stats/ -w
```

### `filter_fasta_from_goa.py`

### `repeat_slurm_job.py`

### `ProtCastDataset2obo.py`

### `swissprot2csv.py`

### `make_dr_seqs.py`

Runs the `mash` application to do pairwise protein sequence comparisons using kmers, then runs
DBSCAN from scikit-learn with the resulting distance data to identify clusters of closely related
sequences that are removed to create a "decreased redundancy" (dr) file. If the input file
is in Swissprot format, the removed sequences will be the ones with the fewest GO terms.

- Feature vector name: There are multiple feature vectors that can be
  generated from a protein sequence such as ctriad, PAAC or SPMAP.

### `scripts/sh`

#### `get_protcast_dataset_files.sh`

Download the 4 input files necessary to build a ProtCastDataset:

```shell
./get_protcast_dataset_files.sh
```

#### `create_protcast_dataset.sh`

## Profiling and Benchmarking

### Tensorflow Profiling

1.The necessary libraries "tensorflow", "tensorrt", and "tensorboard" should all be installed as a part of the pyproject.toml

2.Add a tensorboard callback to the model fitting step to profile your TensorFlow model

```py
          # Profiler callback in binary_classifier.py
          log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
          tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
          
          self.training_model.fit(
              train_tfds,
              epochs=self.epochs,
              validation_data=val_tfds,
              callbacks=[tensorboard_callback]
          )
```

3.Load the relevant modules

```shell
module load all/TensorFlow/2.15.1-Python-3.10 
module load all/CUDA   
```

4.Run the script that fits your model.
   eg. to profile the model in binary_classifier.py you'd run

```py
python3 -t test/data/uniprotkb_gpcrs.fasta -nt test/data/uniprotkb_non-gpcrs.fasta scripts/binary_classify.py
```

### Python Profiling

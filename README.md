# ProtCast

ProtCast is a research codebase for predicting Gene Ontology (GO) annotations
(Molecular Function, Cellular Component, Biological Process) from protein
sequences. The current focus is on **ESM-C protein language model embeddings as
the primary representation**, and on measuring whether augmenting those
embeddings with classical sequence-derived **feature vectors** (AAC, CTriad,
PseKRAAC, etc., from
[protein-feature-vectors](https://github.com/bosborne/protein-feature-vectors))
moves CAFA-standard Fmax / Smin metrics over an ESM-only baseline.

Datasets are built from UniProt/Swiss-Prot, TrEMBL, GO, and UniProt-GOA inputs
and packaged as a `ProtCastDataset`. Multi-label Keras/TensorFlow classifiers
are trained per GO subgraph (by namespace and DAG level).

The three scan workflows that drive the comparison live in `scripts/`:

- `scan_individual_features.py` — ESM-C **+** each individual feature vector
  vs. an ESM-only baseline.
- `scan_feature_vectors_only.py` — each feature vector **alone** (no ESM-C)
  vs. an ESM-only baseline.
- `scan_individual_features.py` invoked with `+`-joined `--algorithms`
  (launched by `scripts/sh/run_scan_combined_features.sh`) — ESM-C +
  **combinations** of feature vectors.

Each scan writes per-algorithm Fmax, Smin, and training time to a JSON file
that is updated incrementally, so interrupted runs resume from where they
left off.

## Installation

### Install *protein-feature-vectors*

[protein-feature-vectors](https://github.com/bosborne/protein-feature-vectors) creates the feature vectors.

```shell
git clone git@github.com:bosborne/protein-feature-vectors.git
cd protein-feature-vectors
pip3 install .
```

### Install `mmseqs`

Binaries can be downloaded at [mmseqs.com](https://dev.mmseqs.com/latest/).

### Install *ProtCast*

```shell
git clone git@github.com:bioteam/protcast.git
cd protcast
pip3 install .
```

## Building the ProtCastDataset

A ProtCastDataset combines protein sequences and GO annotations from multiple input files. A typical
ProtCastDataset has ~0.5M proteins and ~3M GO annotations. The ProtCastDataset is used as input to
TensorFlow for processing and subsequent model-building by Keras.

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

## Repo Organization

├- protcast/ The package directory.

├─ doc/

├─ test/

├─ scripts/

## Running Jobs on TACC for Training and Evaluation

1. SSH into your provisioned Frontera account

2. Request a job queue with GPU access, such as rtx or rtx-dev:

   ```bash
   idev -p rtx-dev -t 00:30:00
   ```

   [TACC documentation](https://docs.tacc.utexas.edu/hpc/frontera/#queues) on queues for reference

3. Load TACC's Apptainer module:

   ```bash
   module load tacc-apptainer
   ```

4. Git Clone ProtCast and its dependencies

5. Start an interactive shell session inside a Singularity container so that it has access to the cluster's NVIDIA GPUs:

   ```bash
   singularity shell --nv tensorflow_2.17.0-gpu.sif
   ```

6. Run the pip install commands for ProtCast and its dependencies if you haven't yet:

   ```bash
   pip3 install .
   ```

7. Now, whatever test or training scripts you run will have access to GPU acceleration

## Profiling and Benchmarking

### Tensorflow Profiling

1. The necessary libraries *tensorflow*, *tensorrt*, and *tensorboard* should all be installed as part of the pyproject.toml

2. Add a tensorboard callback to the model fitting step to profile your TensorFlow model:

   ```python
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

3. Load the relevant modules:

   ```shell
   module load all/TensorFlow/2.15.1-Python-3.10
   module load all/CUDA
   ```

4. Run the script that fits your model. For example, to profile the model in
   `binary_classifier.py` you'd run:

   ```shell
   python3 -t test/data/uniprotkb_gpcrs.fasta \
       -nt test/data/uniprotkb_non-gpcrs.fasta \
       scripts/binary_classify.py
   ```

### Role in Inference

Inference is the process of using a trained model to make a prediction on new, unseen data.

The entire process of inference is a feed-forward pass.

1. You provide an input (e.g., an image or text).
2. The data flows forward through the network's layers.
3. The network produces an output (the prediction).

That's it. There is no other step. So, Inference = Feed-Forward.

### Role in Training

Training is the process of teaching the model by showing it labeled data and updating its weights.

The training process for a single batch of data consists of two main steps:

1. The Feed-Forward Pass: The network takes an input from the training data and performs a forward
   pass to generate a prediction. This is the exact same mechanism as in inference.

2. The Backward Pass (Backpropagation): This is what makes it "training."
    - The model compares its prediction from the forward pass to the correct label and calculates
      the error (loss).
    - It then propagates this error signal backward through the network, calculating the gradient
      (the direction of error change) for every weight.
    - The optimizer uses these gradients to update the weights, slightly improving the network.

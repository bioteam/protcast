# ProtCast

ProtCast is a research codebase for predicting Gene Ontology (GO) annotations
(Molecular Function, Cellular Component, Biological Process) from protein
sequences. The current focus is on **[ESM-C](https://www.evolutionaryscale.ai/blog/esm-cambrian) protein language model embeddings as the primary representation**, and on measuring whether augmenting those
embeddings with classical sequence-derived **feature vectors** (AAC, CTriad,
PseKRAAC, etc., from
[protein-feature-vectors](https://github.com/bioteam/protein-feature-vectors)
increases CAFA-standard Fmax / Smin metrics above an ESM-only baseline.

Datasets are built from UniProt/Swiss-Prot, TrEMBL, GO, and UniProt-GOA inputs
and packaged as a `ProtCastDataset`. Multi-label Keras/TensorFlow classifiers
are trained per GO subgraph (by namespace and DAG level).

Example workflows in `scripts/`:

- `scan_individual_features.py` — ESM-C **+** each individual feature vector
  vs. an ESM-only baseline.
- `scan_feature_vectors_only.py` — each feature vector **alone** (no ESM-C)
  vs. an ESM-only baseline.
- `scan_individual_features.py` invoked with `+`-joined `--algorithms`
  (launched by `scripts/sh/run_scan_combined_features.sh`) — ESM-C +
  **combinations** of feature vectors.
- `compare_knn_esm_vs_knn_combined.py` — two-way KNN comparison of ESM-C
  alone vs. ESM-C concatenated with classical descriptors.

Each scan writes per-algorithm Fmax, Smin, and training time to a JSON file
that is updated incrementally, so interrupted runs resume from where they
left off.

## Installation

### Install *protein-feature-vectors*

[protein-feature-vectors](https://github.com/bioteam/protein-feature-vectors) creates the feature vectors.

```shell
git clone git@github.com:bosborne/protein-feature-vectors.git
cd protein-feature-vectors
pip3 install .
```

### Install *mmseqs*

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

### Results directory convention

Long-running experiments write into `ProtCast_results/` using the
convention `{experiment}-{feature_family}-level-{N}-seed-{S}/`
(e.g. `knn_esm_vs_combined-psekraac-level-6-seed-43`). The sweep
wrappers in [scripts/sh/](scripts/sh/) — `launch_psekraac_sweep.sh`,
`launch_psekraac_sweep_all_levels.sh`,
`launch_psekraac_sweep_all_levels_per_seed.sh` — generate this grid;
matching the naming convention keeps downstream aggregation
(`scripts/sh/run_compare_*_multilevel.sh`, the benchmarking pipeline)
working.

## Configuration (`config.json`)

All training/comparison scripts call `ConfigManager.load_config()` (see
[protcast/config/model_config.py](protcast/config/model_config.py)),
which searches in this order:

1. Path passed explicitly to `load_config(path=...)` (no CLI flag currently
   wired in the scan/compare scripts).
2. `./config.json` in the current working directory.
3. `config.json` next to `model_config.py` (package default).

If none is found a `FileNotFoundError` is raised, so a `config.json`
**must be reachable** when you run a script. `config.json` is gitignored
to keep per-user settings out of the repo — copy
[config.example.json](config.example.json) to `config.json` and edit it:

```bash
cp config.example.json config.json
```

### Example `config.json`

The tracked [config.example.json](config.example.json) is a minimal
config that exercises both the multi-label and KNN code paths without
errors:

```json
{
    "USER": "your_handle",
    "EXPERIMENT_NAME": "your_experiment_name",
    "OPTIMIZER": "adam",
    "LOSS": "binary_crossentropy",
    "METRICS": ["accuracy"],
    "EPOCHS": 100,
    "BATCH_SIZE": 32,
    "HIDDEN_LAYERS": [128, 64],
    "DROPOUT": 0.5,
    "PRED_THRESHOLD": 75.0,
    "VALIDATION_SPLIT": 0.2,
    "PATIENCE": 10,
    "USE_BOX_EMBEDDINGS": false,
    "BOX_DIM": 32,
    "BOX_TEMPERATURE": 10.0,
    "CONTAINMENT_WEIGHT": 0.1,
    "KNN_N_NEIGHBORS": 10,
    "KNN_METRIC": "cosine",
    "KNN_WEIGHTS": "distance",
    "KNN_ALGORITHM": "auto",
    "RANDOM_SEED": 42
}
```

Box-embedding fields (`USE_BOX_EMBEDDINGS`, `BOX_DIM`, `BOX_TEMPERATURE`,
`CONTAINMENT_WEIGHT`) are only consumed when `USE_BOX_EMBEDDINGS: true` —
otherwise the flat multi-label head is used. KNN fields are only consumed
by the KNN code paths. So the same file safely drives every script.

### Running a script

From the project root (`config.json` auto-discovered):

```bash
python3 scripts/scan_individual_features.py \
    -d mf_go_terms-level-8 \
    -p ProtcastDataset.bin \
    -o feature_scan \
    --seed 42 -v
```

From any other directory, either `cd` to a folder that contains a
`config.json` or copy the project `config.json` there first.

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

### TensorFlow Profiling

The `MultiLabelClassifier` already wires up a TensorBoard callback — it's
gated by the `use_tensorboard` constructor argument and writes to
`logs/fit/<timestamp>/` (see
[protcast/model/multilabel_classifier.py:502-506](protcast/model/multilabel_classifier.py#L502-L506)).

1. The required libraries (`tensorflow`, `tensorrt`, `tensorboard`) are pinned
   in `pyproject.toml` and are available inside the
   `tensorflow_2.17.0-gpu.sif` Apptainer container used on Frontera (see
   above).

2. Enable profiling when calling a script that exposes the flag —
   currently `make_multilabel_model_embeds.py`:

   ```bash
   python3 scripts/make_multilabel_model_embeds.py \
       -d mf_go_terms-level-8 \
       -p ProtcastDataset.bin \
       --use_tensorboard
   ```

   To profile from the scan/compare scripts, pass
   `use_tensorboard=True` when instantiating the `MultiLabelClassifier`
   in those scripts (not yet exposed on the CLI).

3. View the run:

   ```bash
   tensorboard --logdir logs/fit
   ```

### Benchmarking Pipeline (`benchmark_pipeline/`)

`benchmark_pipeline/` is a self-contained, post-hoc analysis pipeline
that consumes per-GO-level TSV result files and produces aggregate
metrics, figures, and a summary report. It has its own
`environment.yml` / `requirements.txt` and is independent of the main
`protcast` package (it doesn't import it).

Use it when you want to compare baseline feature-vector methods across
GO DAG levels — efficiency frontiers (F1 vs. cost), head-vs-tail
performance, error correlation between algorithms.

### Quick start

```bash
cd benchmark_pipeline

# 1. Set up the env (conda) and the data/results dir layout
make setup

# 2. Drop one TSV per GO level into data/raw/  (e.g. level_3.tsv, level_4.tsv)
#    Required columns: Algorithm, GO_Term, F1_Score, Sensitivity, Specificity
#    Optional: Vector_Length, Elapsed_Time, TP/TN/FP/FN

# 3. Preprocess + analyze in one shot
make all

# Outputs:
#   results/figures/   efficiency_frontier.png, algorithm_comparison.png, ...
#   results/reports/   benchmark_summary_report.csv, analysis_summary.txt
#   results/data/      enhanced_benchmark_data.csv
```

Pipeline behavior, configurable knobs, and per-script invocation are
documented in
[benchmark_pipeline/PIPELINE_README.md](benchmark_pipeline/PIPELINE_README.md)
and tuned via [benchmark_pipeline/config.yaml](benchmark_pipeline/config.yaml).

## Principles

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

# ProtCast

Extract protein sequences and associated Gene Ontology (GO) annotations from UniProt, TrEMBL and
Gene Ontology Annotation files and use feature vector representations of the protein sequences to
predict the Molecular Function, Cellular Component, and Biological Process GO terms of proteins.
This code uses TensorFlow for structured (tabular) data classification.

## Installation

### Install *protein-feature-vectors*

[protein-feature-vectors](https://github.com/bosborne/protein-feature-vectors) creates the feature vectors.

```shell
git clone git@github.com:bosborne/protein-feature-vectors.git
cd protein-feature-vectors
pip3 install .
```

### Install *ProtCast*

```shell
git clone https://github.com/bioteam/ProtCast
cd ProtCast
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

## Repo Organization

├- protcast/ The package directory.

├─ doc/

├─ test/

├─ scripts/

### `protcast`

#### `protcast/model`

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

3.Load the relevant modules

```shell
module load all/TensorFlow/2.15.1-Python-3.10 
module load all/CUDA   
```

4.Run the script that fits your model.
   eg. to profile the model in binary_classifier.py you'd run

```shell
python3 -t test/data/uniprotkb_gpcrs.fasta -nt test/data/uniprotkb_non-gpcrs.fasta scripts/binary_classify.py
```

### Python Profiling

## Deep Learning

Understanding Feature Transformation in Neural Networks
When we say "the model transforms these features through various operations," we're describing how neural networks progressively convert raw input features into increasingly abstract and task-relevant representations through a series of mathematical transformations.

The Transformation Process

1. Linear Transformations

In artificial neural networks, a neuron (also called a node or unit) is the fundamental computational unit that:

Receives multiple input signals
Applies a mathematical transformation to these inputs
Produces a single output value
Anatomical Structure of an Artificial Neuron
Components
Inputs: Values received from previous layer neurons or raw features
Weights: Learnable parameters that determine the importance of each input
Bias: A learnable offset parameter
Aggregation Function: Typically a weighted sum of inputs
Activation Function: Non-linear function applied to the aggregation result
Output: The signal sent to the next layer

Based on Role
Input Neuron: Passes input features to the first hidden layer
Hidden Neuron: Located in middle layers, performs intermediate computations
Output Neuron: Produces final predictions
Specialized Neurons
Convolutional Neuron: Applies filter operations on structured data like images
LSTM/GRU Cell: Complex neurons with memory capabilities for sequence data
Attention Unit: Computes relevance weights between different elements
Pooling Unit: Performs downsampling operations

Practical Significance
The neuron is the basic building block of neural networks. By combining thousands or millions of these simple units in different architectures (feedforward, convolutional, recurrent, etc.), neural networks can learn complex patterns and perform sophisticated tasks.

The power of neural networks emerges not from the complexity of individual neurons, but from their collective behavior when arranged in layers and trained with optimization algorithms like gradient descent.

A neuron is not merely a method (function) but rather a computational unit with:

State: Contains learnable parameters (weights and bias)
Behavior: Performs a specific mathematical operation
Connectivity: Links to other neurons in a network architecture

While we conceptualize individual neurons, modern frameworks optimize by:

Vectorizing Operations: Processing entire layers at once
Batch Processing: Handling multiple samples simultaneously
GPU Acceleration: Organizing computations for parallel execution
This creates a distinction between:

The conceptual neuron (a single computational unit)
The implemented neuron (part of an optimized layer operation)

Each neuron applies a weighted sum to its inputs:

z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

Where:

x₁, x₂, ..., xₙ are input features or outputs from previous layers
w₁, w₂, ..., wₙ are learned weights
b is a learned bias term
In Protein Context: This might combine various protein features (hydrophobicity, charge, etc.) with different importance weights.

2. Non-linear Activations
The weighted sum is passed through a non-linear function:

a = activation_function(z)
Common activation functions:

ReLU: max(0, z) (keeps positive values, zeros out negatives)
Sigmoid: 1/(1+e^(-z)) (squashes values between 0 and 1)
Tanh: (e^z - e^(-z))/(e^z + e^(-z)) (squashes values between -1 and 1)

In Protein Context: This introduces non-linearity, allowing the model to capture complex patterns in protein features.

3. Hierarchical Feature Extraction
As data flows through multiple layers:

First layers: Capture simple combinations of raw features
Middle layers: Build intermediate abstractions
Deep layers: Form high-level concepts relevant to the task
In Protein Context: Early layers might identify simple amino acid patterns, while deeper layers could recognize functional domains or structural motifs.

Concrete Example with Protein Data
Let's trace how protein features transform through a simple network:

Input Layer
Input: [0.7, 0.2, 0.9, 0.1, 0.5]  # Protein features (e.g., hydrophobicity, charge...)
First Hidden Layer (4 neurons)
For the first neuron:

z₁₁ = (0.7 × 0.3) + (0.2 × -0.1) + (0.9 × 0.4) + (0.1 × 0.7) + (0.5 × 0.2) + 0.1
    = 0.21 - 0.02 + 0.36 + 0.07 + 0.1 + 0.1
    = 0.82

a₁₁ = ReLU(0.82) = 0.82  # Apply activation
Similarly for other neurons, producing:

First layer output: [0.82, 0.56, 0.0, 0.93]
Transformation: Raw protein features → Feature combinations

Second Hidden Layer (3 neurons)
Taking the first layer output as input:

z₂₁ = (0.82 × 0.5) + (0.56 × 0.7) + (0.0 × 0.2) + (0.93 × -0.3) + 0.2
    = 0.41 + 0.392 + 0.0 - 0.279 + 0.2
    = 0.723

a₂₁ = ReLU(0.723) = 0.723
Calculating for all neurons:

Second layer output: [0.723, 1.104, 0.0]
Transformation: Feature combinations → Higher-level patterns

Output Layer (1 neuron for binary classification)
z₃₁ = (0.723 × 0.8) + (1.104 × 0.6) + (0.0 × 0.3) + 0.1
    = 0.5784 + 0.6624 + 0.0 + 0.1
    = 1.3408

a₃₁ = Sigmoid(1.3408) = 0.7925  # Probability of having the GO term
Transformation: Higher-level patterns → Classification probability

What's Actually Happening Biologically
These mathematical transformations are detecting meaningful biological patterns:

First Layer: May identify basic physicochemical patterns (e.g., hydrophobic regions, charged clusters)

Middle Layers: Could recognize sequence motifs, secondary structure elements, or binding sites

Deep Layers: Might capture complex functional domains, interaction sites, or structural configurations relevant to the GO function

Output Layer: Integrates these detected patterns to predict whether the protein performs the specific function defined by the GO term

Through this progressive transformation process, the network converts raw protein features into increasingly sophisticated representations that ultimately enable accurate GO term prediction.

What Determines the Number of Neurons in a Layer
The number of neurons in a layer is one of the most important architectural decisions in neural network design, influenced by multiple factors:

Key Determining Factors

1. Problem Complexity
More Complex Problems → More Neurons
Tasks with intricate patterns require more representation capacity
Higher-dimensional data typically needs more neurons to process
Subtle differences between classes may require more neurons to distinguish
2. Input/Output Dimensionality
Input Layer: Typically matches the number of input features
Output Layer: Determined by the task type:
Classification: Typically one neuron per class (or a single neuron for binary classification)
Regression: Usually matches the dimension of the target variable(s)
Generation: Depends on the output structure required
3. Available Training Data
More Data → Can Support More Neurons
Larger datasets can train more complex models without overfitting
Limited data requires more conservative neuron counts
Rule of thumb: Number of training samples should exceed number of parameters
4. Computational Constraints
Limited Resources → Fewer Neurons
Training time increases with more neurons
Memory requirements scale with layer size
Inference speed constraints may limit neuron count

How Neuron Weights Change During Training
Yes, Weights Change Throughout Training
The core principle of neural network training is the systematic adjustment of weights (and biases) to minimize prediction errors. This is the fundamental learning process.

Weight Update Mechanism

1. Initial State
Random Initialization: Weights typically start with small random values

# Common initialization in code

weights = np.random.normal(0, 0.01, size=(input_size, output_size))
Purpose: Random initialization breaks symmetry and enables diverse feature detection
2. Forward Pass
Input data passes through the network
Current weights determine the prediction
Error (loss) is calculated between prediction and actual target
3. Backward Pass (Backpropagation)
Gradient Calculation: The system calculates how much each weight contributed to the error
Chain Rule Application: Error gradients flow backward through the network
Result: Each weight receives a gradient indicating how it should change
4. Weight Update
Update Rule: Weights are adjusted in the direction that reduces error
new_weight = current_weight - learning_rate * gradient
Learning Rate: Controls the size of weight updates (typically 0.001-0.01)
Visual Example of Weight Evolution
Consider a single neuron with two inputs tracking its weights during training:

Epoch 0: weights = [0.08, -0.12]  (random initialization)
Epoch 10: weights = [0.21, -0.19]  (early adjustments)
Epoch 50: weights = [0.45, -0.37]  (significant changes)
Epoch 100: weights = [0.52, -0.41]  (refinement)
Epoch 200: weights = [0.53, -0.42]  (stabilization)
Characteristics of Weight Changes

1. Learning Phases
Early Training: Large, rapid weight changes
Middle Training: Moderate, focused adjustments
Late Training: Small, fine-tuning refinements
Convergence: Minimal changes as weights stabilize
2. Layer-Specific Patterns
Early Layers: Typically learn faster and stabilize earlier
Deep Layers: Often take longer to converge
Output Layer: Can show more volatile changes
3. Weight Magnitude Evolution
Many weights grow from small random values to larger magnitudes
Some weights may approach zero (especially with regularization)
Certain critical weights might become very large (detecting key features)
Factors Affecting Weight Changes
1. Learning Rate
High Learning Rate: Larger, more rapid weight updates (risk of instability)
Low Learning Rate: Smaller, more stable updates (risk of slow convergence)
Learning Rate Schedules: Strategically reducing learning rate over time
2. Optimizer Choice
SGD: Simple updates proportional to the gradient
Adam/RMSprop: Adaptive updates based on historical gradient information
Momentum-based: Incorporates previous update directions
3. Regularization
L1/L2 Regularization: Adds penalties that push weights toward zero
Dropout: Temporarily disables neurons, affecting weight update patterns
Batch Normalization: Changes the dynamics of weight updates
Code Example: Visualizing Weight Changes

```py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Function to extract weights
def get_weights(model):
    return [layer.get_weights()[0].flatten() for layer in model.layers if len(layer.get_weights()) > 0]

# Track weights during training
weight_history = []
initial_weights = get_weights(model)

# Custom callback to record weights
class WeightRecorder(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Record every 5 epochs
            weight_history.append(get_weights(model))

# Train model with callback
model.fit(X_train, y_train, epochs=100, callbacks=[WeightRecorder()], verbose=0)

# Plot weight evolution
plt.figure(figsize=(10, 6))
for i, layer_weights in enumerate(zip(*weight_history)):
    weights_array = np.array(layer_weights)
    for j in range(weights_array.shape[1]):
        plt.plot(weights_array[:, j], label=f'Layer {i+1}, Weight {j+1}')

plt.xlabel('Epoch (×5)')
plt.ylabel('Weight Value')
plt.title('Evolution of Neural Network Weights During Training')
plt.legend()
plt.grid(True)
plt.show()
```

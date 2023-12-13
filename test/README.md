# Tests

This directory contains some tests around ProtCast data processing scripts

`path_to_root.py`: Tests the generation of the Ontology from and `.obo` file.
It takes a serialized Ontology file, a GO term namespace, and a GO term ID and
prints out the path from the input GO term to the root of the tree. Sample
usage:

```
python3 test/path_to_root.py data/ontology/serialized/go.bin GO:0000102
```

`ontology_topology.py`: Tests that the GO term levels are correctly populationg using topological
sorting criteria and that the GOTerms have the correct set of ancestors.

```
python3 test/ontology_topology.py data/ontology/serialized/go.bin
```

`annot_propagation.py`: Tests that a protein if a protein is annotated with a GO term it is
also annotated with all of its ancestors. Usage:
```
python3 test/annot_propagation.py data/dataset/dataset.bin
```
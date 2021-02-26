# Standard project: Tagger–parser pipeline

## Baseline

The baseline for the standard project is a simple pipeline architecture with the following components:
- a part-of-speech tagger
- a dependency parser
- code to read and output dependency trees in the CoNLL-U format

It is possible to train and evaluated the system on any given Universal Dependencies treebank. Some of the Universal Dependencies treebanks contain so-called non-projective trees. To train on these treebanks, we first projectivize them.  Functions have been put in separate files. The main script used for training and evaluating the system is found in the notebook. The notebook reports the tagging accuracy and unlabelled attachment score when trained on the training sections and evaluated on the development sections of the English Web Treebank (EWT). In order to replicate the results, simply run each cell in the notebook.


## Project Work 

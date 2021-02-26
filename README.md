# Standard project: Tagger–parser pipeline

## Baseline

The baseline for the standard project is a simple pipeline architecture with the following components:
- a part-of-speech tagger
- a dependency parser
- code to read and output dependency trees in the CoNLL-U format

It is possible to train and evaluated the system on any given Universal Dependencies treebank. Some of the Universal Dependencies treebanks contain so-called non-projective trees. To train on these treebanks, we first projectivize them.  Functions have been put in separate files. The main script used for training and evaluating the system is found in the notebook. The notebook reports the tagging accuracy and unlabelled attachment score when trained on the training sections and evaluated on the development sections of the English Web Treebank (EWT). In order to replicate the results, simply run each cell in the notebook.

## Project Work 

During the two project weeks (W9–W10), you will extend and/or apply your baseline system according to your project plan. At the end of this period, you must submit a one-paragraph abstract for your project. The abstract should summarize what you have actually done in the project (which may be different from what you planned to do), as well as your main results. The main purpose of the abstract is to announce your presentation ahead of the final ‘conference’ that will take place in W11.

**Deliverable:** Submit a plain text file containing the following: (a) a one-paragraph abstract of your project (no longer than 200 words), and (b) a link to a GitLab repository containing your code.

**Due date:** 2021-03-12

# Sequence-to-sequence workflows: Named Entity Recognition Project

NER project for the NLP course Méthode Empirique en Traitement de Langage at UNIGE

In this assignment we will put together various steps involved in sequence-to-sequence workflow, from accessing the data to evaluating the output of a system

- Establish the baseline performance with a pre-neural tool
- Establish the baseline performance with a neural tool
- Analyse system outputs
- Propose and evaluate improvements over both baselines

_*Data set*_ : CoNLL2003 

_Pre-neural tool_ : crfsuite.


### Models

- Neural Baseline: Transformer based, taken from the [following tutorial](https://keras.io/examples/nlp/ner_transformers/)
- CRF baseline: using [scikit-learn library](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#let-s-use-conll-2002-data-to-build-a-ner-system)
- Neural baseline model improvement tentative
- BiLSTM model
- BiLSTM model with an extra CRF Layer

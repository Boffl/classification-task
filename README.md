# Training Classifiers

This project is part of a larger project for a language based game, 
which requires classification of texts. The models themselves, are kept on the local directory
beause of their size


## Bert Classifier

A fine-tuned BERT model, with a one layer classifier 
on top. The initial BERT is the distilbert-base-uncased. The model will eventually
be trained on data from the game, but until then a news dataset from
the bbc[^1] is used. It can be found [here](http://mlg.ucd.ie/datasets/bbc.html). <br>

### Scripts
`BERT_for_multiple_classification_problem.ipynb` contains the fine-tuning of the model. <br>
`Loading_BERT_and_evaluate.ipynb` loads the model and evaluates its performance
on the test set. For the test set the model achieves a microF1 of 0.975. <br>
`run_BERT_classifier.py` is a script that runs the model's
prediction on a random datapoint from the test set and times
different parts of the code. <br>
`run_BERT_classifier_onnx.py` contains code for a more effective implementation
of the prediction. At its heart is the function `inference`, which
runs a forward pass on a model saved in onnx format, given an input string. It returns the resulting logits as a numpy array.


[^1]: D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.

## GPT-3 Classifier
We tested the possibility of making a classifier with
the GPT-3 API, by setting the logit_bias of the labels to 100.
This essentially means that we ask the model to generate the 
next word, while giving a high prior probability to the labels
that we want to predict.
To do this we need a dictionary with the GPT tokens for the labels. This is generated
with the code found in this folder.
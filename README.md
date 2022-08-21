# Training Classifiers

This project is part of a larger project for a language based game, 
which requires classification of texts. The models will eventually
be trained on data from the game, but until then a news dataset from
the bbc[^1] is used. It can be found [here](http://mlg.ucd.ie/datasets/bbc.html). <br>
----
[^1]: D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.

## Bert Classifier
A fine-tuned BERT model, with a one layer classifier 
on top. The initial BERT is the bert-base-uncased.
###Scripts
The two jupyter notebooks were run on Google Colab. `BERT_for_multiple_classification_problem.ipynb`
contains the original fine-tuning of the model.
`Loading_BERT_and_evaluate.ipynb` loads the model and evaluates its performance
on the test set. For the test set the model achieves a microF1 of 0.973. <br>
`run_BERT_classifier.py` is a script that runs the models
prediction on a random datapoint from the test set and times
different parts of the code.

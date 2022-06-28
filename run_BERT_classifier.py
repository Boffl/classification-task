#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Nikolaj Bauer

import numpy as np
import pandas as pd
import torch
import random
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import time

time_0 = time.perf_counter()  # measure execution time

# setting up the device
# Need this for setting up the variable device, even when not on a gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_input_data(filename, n) -> (str, str):
    """
    Gets a text and its label from the test dataset for the evaluate_example funcion
    @param  filename(str): path to a csv file with two columns: 'category' and 'text'
    @param  n(int): number of rows in the file
    @return tuple(label, text)"""

    # choosing a random line in order not to read in the whole file
    s = 1  # sample size
    # the 0-indexed header will not be included in the skip list, to include the header
    skip = sorted(random.sample(range(1, n + 1), n - s))
    example = pd.read_csv(filename, skiprows=skip)

    # # Read in whole file and take a sample
    # data_test = pd.read_csv("filename")
    # example = data_test.sample(1)

    # return the data as strings
    # note: .values gives a list of the values, here a list with one element
    return example.category.values[0], example.text.values[0]

# define the classes and encode
news_classes = ['entertainment', 'politics', 'tech', 'sport', 'business']
num_labels_news = len(news_classes)
le_news = LabelEncoder()
le_news.fit(news_classes)

# get the pretrained tokenizer, which now is saved to ./tokenizer
time_tokenizer_1 = time.perf_counter()
tokenizer = BertTokenizer.from_pretrained("./tokenizer")
time_tokenizer_2 = time.perf_counter()


def preprocessing_for_bert(data):
    """Perform required preprocessing steps for BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # cut too long sentences, only take the end

        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=512,  # Max length to truncate/pad
            padding = 'max_length',  # Pad sentence to max length
            truncation=True,
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False, num_labels=5):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, num_labels

        # Instantiate BERT model
        # here we do not need to download the pretrained version, since we are using our fine-tuned model
        bert_config = BertConfig() # Configuration (i.e. some hyperparameters) for bert-base-uncased
        self.bert = BertModel(bert_config)

        # one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

# load the fine-tuned model
time_load_model_1 = time.perf_counter()
news_model = BertClassifier(num_labels=num_labels_news)
news_model.load_state_dict(torch.load("BERT_classifier_news_data.pt", map_location=device))
time_load_model_2 = time.perf_counter()
# note to myself: map_location=device is important if using cpu, the default is cuda
# the device varialbe is defined above in Set up GPU for training.


def make_prediction(model:BertClassifier, le:LabelEncoder, texts:list[str]):
    """Make a prediction given a finetuned BertClassifier, a LabelEncoder and a string
     @param: model(BertClassifier): a BertClassifier object from the Transformers library
     @param: le(LabelEncoder): Label encoder from sklearn, already fitted to the labels of the dataset
     @param: text(list[str]): a list of texts to be classified
     @return: A string with the predicted label (decoded with the labelencoder)"""

    model.eval()
    # Run `preprocessing_for_bert` on the test set, get embeddings and masks
    test_inputs, test_masks = preprocessing_for_bert(texts)  # takes a list input

    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=16)

    all_logits = []
    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()


    # Get predictions from the probabilities
    preds = [np.argmax(prob) for prob in probs]

    # Get the label from the probs:
    return le.inverse_transform(preds)


def evaluate_on_test_set():
    data_test = pd.read_csv("bbc-text_test.csv")
    probs = make_prediction(news_model, le_news, data_test.text)
    print(probs)



def evaluate_example():
    """Evaluate on one example and measure the time"""
    time_1 = time.perf_counter()
    # Get the data
    time_load_data_1 = time.perf_counter()
    label, text = get_input_data("bbc-text_test.csv", 445)
    time_load_data_2 = time.perf_counter()

    time_predict_1 = time.perf_counter()
    # since we input only one sentence take the first (and only) element
    prediction = make_prediction(news_model, le_news, [text])[0]
    time_predict_2 = time.perf_counter()

    time_2 = time.perf_counter()
    print(f"Input Text: \n{text}\n")
    print(f"total time elapsed: {round(time_2 - time_0, 4)} seconds")
    print(f"time to load the text: {round(time_load_data_2 - time_load_data_1, 4)}")
    print(f"time to load tokenizer: {round(time_tokenizer_2 - time_tokenizer_1, 4)}")
    print(f"time to load the model: {round(time_load_model_2 - time_load_model_1, 4)}")
    print(f"time to make predicion: {round(time_predict_2 - time_predict_1, 4)}")
    print(f"\n------------------------\nCorrect label: {label}")
    print(f"Predicted label: {prediction}")



if __name__ == "__main__":
    evaluate_example()
    # evaluate_on_test_set()


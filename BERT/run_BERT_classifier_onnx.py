#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Nikolaj Bauer

import sys
import os

import time
import numpy as np
# import math  # if we one day can get around using numpy...
from transformers import DistilBertTokenizer as BertTokenizer
import onnxruntime as ort
# from sklearn.preprocessing import LabelEncoder

directory = os.path.dirname(__file__)
sys.path.append(directory)
models_path = os.path.join(directory, "distilBert_news_model.onnx")
tokenizer_path = os.path.join(directory, 'tokenizer')


def softmax(x):
    """calculate the softmax of an array
    @:param x: nparray or list with numeric values
    @:return: numpy.ndarray"""
    return np.exp(x)/sum(np.exp(x))

# def softmax(x):
#     exponents = []
#     for element in x:
#         exponents.append(math.exp(element))
#     summ = sum(exponents)
#     for i in range(len(exponents)):
#         exponents[i] = exponents[i] / summ
#     return exponents

class LabelEncoder():
    """Simple label encoder so I do not have to import the whole sklearn package.
    For our case it should behave exactly the same as sklearn.preprocessing.LabelEncoder"""
    def __init__(self, labels:list):
        self.labels = sorted(labels)

    def encode(self, element: str) -> int:
        return self.labels.index(element)

    def decode(self, idx: int) -> str:
        return self.labels[idx]


tokenizer = BertTokenizer.from_pretrained(tokenizer_path)


def preprocessing_for_bert(data):
    """Perform required preprocessing steps for BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (list[list]): List of token ids for each input sentence in data
    @return   attention_masks (list[list]): List of attention masks for each input sentence in data
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:

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

    # we do not even need nd.arrays
    # Convert lists to ndarrays (since we are using onnx we do not need tensors)
    # input_ids = np.asarray(input_ids, dtype=np.int64)
    # attention_masks = np.asarray(attention_masks, dtype=np.int64)

    return input_ids, attention_masks


def inference(text, model=models_path):
    """perform a forward pass on a model that is saved as onnx
    :param model: a filepath to a .onnx file
    :param text: a string to be passed to the model
    :return: the logits as a numpy array"""

    ort_session = ort.InferenceSession(model)

    inputs = preprocessing_for_bert([text])
    outputs = ort_session.run(
        None,
        {"input_ids": inputs[0], "input_masks":inputs[1]},
    )

    return outputs[0][0]


def get_prediction(logits, le):
    """Transform the output of the model into a prediction
    :param logits: a numpy array, the output of a forward pass
    :param le: a LabelEncoder that is used to decode the label
    :return: string of the label that the model predicts"""

    probs = softmax(logits)
    pred = np.argmax(probs)
    return le.decode(pred)

    # with the sklearn LabelEncoder:
    # return le.inverse_transform([pred])[0]


if __name__=="__main__":

    news_classes = ['entertainment', 'politics', 'tech', 'sport', 'business']
    num_labels_news = len(news_classes)
    le_news = LabelEncoder()
    le_news.fit(news_classes)

    label, text = ("business", "This is an article about the FED raising interest rates in the upcoming quater. Goldman Sachs executives criticized this move.")
    print("start inference...")
    time_0 = time.perf_counter()
    logits = inference(text, models_path)
    time_1 = time.perf_counter()

    prediction = get_prediction(logits, le_news)
    print(f"correct label: {label}\npredicted label: {prediction}\n"
          f"time used for inference: {time_1-time_0}")


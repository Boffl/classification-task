#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Nikolaj Bauer

import time
importtime_0 = time.perf_counter()
import torch
import numpy as np
from transformers import BertTokenizer
import onnxruntime as ort
from sklearn.preprocessing import LabelEncoder
import os
importtime_1 = time.perf_counter()

# only use this while testing on existing dataset...
# takes 3.3 seconds to import...
from run_BERT_classifier import get_input_data







def softmax(x):
    return np.exp(x)/sum(np.exp(x))


tokenizer = BertTokenizer.from_pretrained("./tokenizer")
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




def inference(model, text):
    time_start_sess_0 = time.perf_counter()
    ort_session = ort.InferenceSession(model)
    time_start_sess_1 = time.perf_counter()
    print(f"time to start inf-sess: {time_start_sess_1-time_start_sess_0}")

    time_start_pred_0 = time.perf_counter()
    inputs = preprocessing_for_bert([text])
    outputs = ort_session.run(
        None,
        {"input_ids": inputs[0].numpy(), "input_masks":inputs[1].numpy()},
    )
    time_start_pred_1 = time.perf_counter()
    print(f"time to start predict: {time_start_pred_1 - time_start_pred_0}")

    return outputs[0][0]

def get_prediction(logits, le):
    # print(outputs)  # list with one element: Numpy array of the logits
    probs = softmax(logits)
    pred = np.argmax(probs)
    return le.inverse_transform([pred])[0]

if __name__=="__main__":
    print(f"importing time: {importtime_1 - importtime_0}")
    news_classes = ['entertainment', 'politics', 'tech', 'sport', 'business']
    num_labels_news = len(news_classes)
    le_news = LabelEncoder()
    le_news.fit(news_classes)

    # only for testing the news dataset, in the End we should just put a string as input
    label, text = get_input_data("data/bbc-text_test.csv", 445)
    print("start inference...")
    time_0 = time.perf_counter()
    model = os.path.abspath(r"C:\Users\nik_b\Documents\UZH\Party_Game\models\news_model.onnx")
    logits = inference(model, text)
    time_1 = time.perf_counter()

    prediction = get_prediction(logits, le_news)
    print(f"correct label: {label}\npredicted label: {prediction}\n"
          f"time used for inference: {time_1-time_0}")


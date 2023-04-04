from sklearn.metrics import f1_score, classification_report
import pandas as pd
import re

gpt_labels = []
real_labels = []

with open("GPT-3_labels.csv", "r") as gpt_file:
    with open("action_data_csv.csv", "r") as real_file:
        real_file.readline() # skip the header
        for gpt_line in gpt_file:
            real_line = real_file.readline()
            real_label = re.match("(.+?),", real_line).group(1)
            gpt_label = re.match("(.+?);", gpt_line).group(1)
            gpt_labels.append(gpt_label)
            real_labels.append(real_label)

            real_text = re.match(".+?,(.+)", real_line).group(1)
            gpt_text = re.match(".+?;(.+)", gpt_line).group(1)

# f1 micro:
print(len(real_labels), len(gpt_labels))

print(classification_report(real_labels, gpt_labels))
print("f1 micro: ", f1_score(real_labels, gpt_labels, average="micro"))
print("f1 macro: ", f1_score(real_labels, gpt_labels, average="macro"))
import openai
import os
import re
from tqdm import tqdm

openai.api_key = os.getenv("API_KEY")
model = "text-davinci-002"

logit_bias = {'4098': 100, '4483': 100, '270': 100, '1561': 100, '676': 100, '39463': 100, '296': 100, '5548': 100, '4144': 100, '47408': 100, '9280': 100, '67': 100, '7109': 100, '198': 100, '590': 100, '466': 100, '85': 100, '2147': 100, '2666': 100, '16620': 100, '32638': 100, '4598': 100, '1660': 100, '44542': 100}
text = "wanna dance? Answer: nope"

parameters = {
    "model": "text-davinci-002",
    "prompt": text,
    "temperature": 0.9,
    "max_tokens": 4,
    "frequency_penalty": 2,
    "presence_penalty": 0.2,
    "logit_bias": logit_bias,
    "stop": ["\""]
}


with open("GPT-3_labels.csv", "w", encoding="utf-8") as outfile:
    with open("action_data_csv.csv", "r", encoding="utf-8") as infile:
        infile.readline()  # jump the header line
        for i, line in tqdm(enumerate(infile), total=500):
            #Todo: not all lines match here, find out which ones
            # Some of them (the ones by Jean) have a different format..... So we only go until 349
            input = re.match(r'.*,\s?(""".*|)', line)  # matching the conversation or an empty string (some lines are empty)
            if not input:
                print("Problem on line: ", i)
                break

            input = input.group(1)
            try:
                name = re.search(r'""\\n(\w+):',line).group(1)
            except AttributeError:
                print("Problem on line: ", i)
                print(line)

            text = line + f"after this convesation {name} went to "

            parameters = {
                "model": "text-davinci-002",
                "prompt": text,
                "temperature": 0.9,
                "max_tokens": 4,
                "frequency_penalty": 2,
                "presence_penalty": 0.2,
                "logit_bias": logit_bias,
                "stop": ["\""]
            }
            response = openai.Completion.create(**parameters)
            label = response["choices"][0]["text"].strip()
            # !!! watch out, for the null action we here will have "do nothing", when comparing we have to change it back
            # The way it is set now, is that if the model makes a label that is not in there, we just put none there
            if label not in ["drink water", "drink alcohol", "eat", "dance", "leave", "vomit", "talk", "pee"]:
                label = "Null"

            outfile.write(f"{label}; {input}\n")



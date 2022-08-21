from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("./tokenizer")

def tokenize_labels(labels: list[str]) -> list[str]:
    """
    Converts a list of labels (or a stream) into a list of GPT-3 tokens.
    Adds preceding whitespace as needed in order to account for
    quirks in how GPT-3 handles tokenization.
    """
    # Start with whitespace tokens
    tokens = []
    # Tokenize each label by itself *and* with a preceding space.
    for label in labels:
        tokens += tokenizer.encode(label)
        tokens += tokenizer.encode(" " + label)

    return tokens

def get_logit_bias(labels: list[str]) -> dict[str, float]:
    """
    Returns a logit_bias that can be used to constrain GPT-3
    predictions to a set of pre-determined character sequences
    (i.e. phrases or words). Intended to be used for classification
    problems.
    """
    tokens = tokenize_labels(labels)
    logit_bias: dict[str, float] = {}
    for token in set(tokens):
        # Set the logit_bias for each token to 100, effectively
        # forcing GPT-3 to only choose from these tokens.
        logit_bias[str(token)] = 100
    return logit_bias

with open("labels.txt", "r", encoding="utf-8") as f:
    print(type(f))
    logit_bias = get_logit_bias(f)
    print(logit_bias)
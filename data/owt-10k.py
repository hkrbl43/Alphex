import os
import requests
import tiktoken
import numpy as np

from datasets import load_dataset
dataset_name = "stas/openwebtext-10k"
name = dataset_name.split('/')[-1]
ds = load_dataset(dataset_name, split='train')
enc = tiktoken.get_encoding("gpt2")
tokenized_ds = enc.encode_ordinary(ds["text"])

# encode with tiktoken gpt2 bpe
validation_fraction = 0.1

# Calculate the number of examples for validation
num_validation_examples = int(len(tokenized_ds) * validation_fraction)

# Split the dataset into training and validation sets
train_dataset = tokenized_ds[:-num_validation_examples]
validation_dataset = tokenized_ds[-num_validation_examples:]

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile('train.bin')
val_ids.tofile('val.bin')

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens

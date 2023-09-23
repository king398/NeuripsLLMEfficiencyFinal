import datasets
from datasets import Dataset, DatasetDict

import glob

cnn_dailymail = datasets.load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/cnn_dailymail_2_0")
gsm8k = datasets.load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/gsm8k"
)
openbookqa = datasets.load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/openbookqa"
)
texts = cnn_dailymail['prompt'] + gsm8k['prompt'] + openbookqa['prompt']

data = {"prompt": texts}

# Create a Dataset from this data
dataset = Dataset.from_dict(data)

# If you want to wrap it in a DatasetDict
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/all_prompts")

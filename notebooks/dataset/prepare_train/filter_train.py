import datasets
from datasets import Dataset, DatasetDict

import glob

truthful_qa = datasets.load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/truthful_qa")
mmlu = datasets.load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/mmlu_test")
cnn_dailymail = datasets.load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/cnn_dailymail_2_0")
bbq = datasets.load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/BBQ")
texts = mmlu['prompt'] + cnn_dailymail['prompt'] + bbq['prompt'] + truthful_qa['prompt']

data = {"prompt": texts}

# Create a Dataset from this data
dataset = Dataset.from_dict(data)

# If you want to wrap it in a DatasetDict
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/all_prompts")

import datasets
from transformers import AutoTokenizer, AutoModel
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np1

dataset = datasets.load_dataset("cnn_dailymail", "2.0.0")['train']
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(int(len(dataset) * 0.03)))


def make_prompt(example):
    prompt = f"""Document:{example['article']}
    Summary: {example['article']}"""
    return {"prompt": prompt}


dataset = dataset.map(make_prompt)
dataset.save_to_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/cnn_dailymail_2_0")

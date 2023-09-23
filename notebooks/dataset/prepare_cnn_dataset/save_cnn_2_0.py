import datasets
from transformers import AutoTokenizer, AutoModel
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np1

dataset = datasets.load_dataset("cnn_dailymail", "2.0.0")
print(dataset['train'][0])
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(torch.device('cuda'))
model.eval()
model = torch.nn.DataParallel(model)

from functools import partial


def get_embeddings(texts):
    embeddings = []
    batch_size = 256
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]

        # Tokenize the texts in the batch all at once
        inputs = tokenizer(batch_texts, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        inputs = {k: v.to(torch.device("cuda")) for k, v in inputs.items()}

        with torch.no_grad():
            with autocast():
                outputs = model(**inputs)
                outputs = outputs.pooler_output
        embeddings.append(outputs)

    embeddings = torch.cat(embeddings, dim=0)
    print(embeddings.shape)
    return embeddings


text = dataset['train']['article']

embeddings = get_embeddings(text)
embeddings_np = embeddings.cpu().numpy()

# Using NearestNeighbors to find the closest neighbors (most similar items)
knn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1)
knn.fit(embeddings_np)

# Initialize an empty list to collect the distances
collected_distances = []

# Define batch size for querying the index
query_batch_size = 256

# Loop through embeddings in batches and query the index
for i in tqdm(range(0, len(embeddings_np), query_batch_size)):
    batch_query = embeddings_np[i:i + query_batch_size]
    batch_distances, _ = knn.kneighbors(batch_query, 2)

    # Convert cosine similarities to cosine distances
    batch_distances = 1 - batch_distances

    collected_distances.append(batch_distances)

# Concatenate all the distances
distances = np.concatenate(collected_distances, axis=0)


def add_max_cosine_similarity(example, distances):
    example['max_cosine_similarity'] = 1 - distances[example['index'], 1]
    return example


# Add an index column to the dataset for tracking
dataset['train'] = dataset['train'].add_column('index', list(range(len(dataset['train']))))

# Use the map function to add the max_cosine_similarity
dataset['train'] = dataset['train'].map(partial(add_max_cosine_similarity, distances=distances))


def filter_by_similarity(example):
    return example['max_cosine_similarity'] < 0.95


# Apply the filter function
filtered_dataset = dataset['train'].filter(filter_by_similarity)


def make_prompt(example):
    prompt = f"""Document:{example['article']}
    Summary: {example['article']}"""
    return {"prompt": prompt}


filtered_dataset = filtered_dataset.map(make_prompt)

filtered_dataset.save_to_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/cnn_dailymail_2_0")

import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(torch.device('cuda'))
model.eval()
model = torch.nn.DataParallel(model)
df = pd.read_csv('/home/mithil/PycharmProjects/NeuripsLLM/data/train_bigbench.csv')
df = df.sample(1000)

def get_embeddings(texts):
    embeddings = []
    batch_size = 256
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]

        # Tokenize the texts in the batch all at once
        inputs = tokenizer(batch_texts, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        inputs = {k: v.to(torch.device("cuda")) for k, v in inputs.items()}

        with torch.no_grad() :
            with autocast():
                outputs = model(**inputs)
                outputs = outputs.pooler_output
        embeddings.append(outputs)

    embeddings = torch.cat(embeddings, dim=0)
    print(embeddings.shape)
    return embeddings

def return_prompt(example):
    prompt = f"""The following are multiple choice questions . Please choose the correct answer from the four choices
    Question: {example['inputs']}
    Options: {example['multiple_choice_targets']}
    targets. {example['targets']}"""
    return {'output_text': prompt}


df['output_text'] = df.apply(lambda row: return_prompt(row)['output_text'], axis=1)
output_text = df['output_text']
batch_size = 1

embeddings = get_embeddings(output_text.to_list()   )
embeddings_np = embeddings.cpu().numpy()

# Using NearestNeighbors to find the closest neighbors (most similar items)
knn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1)  # n_neighbors=2 to include the point itself
knn.fit(embeddings_np)
distances, _ = knn.kneighbors(embeddings_np)

# Since the point itself is its own nearest neighbor, it will have a distance of 0.
# Thus, we'll retrieve the second smallest distance, which corresponds to the closest point.
df['max_cosine_similarity'] = 1 - distances[:, 1]  # converting distance to similarity

print(df.head())

print(df.head())

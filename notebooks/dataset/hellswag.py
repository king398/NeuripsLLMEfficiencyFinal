import datasets

dataset = datasets.load_dataset("Rowan/hellaswag")['train']
print(dataset[0]['endings'])


def make_prompt(example):
    prompt = ""
    return {"prompt": prompt}


dataset = dataset.map(make_prompt)
dataset = dataset.shuffle(seed=42)
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/hellaswag")

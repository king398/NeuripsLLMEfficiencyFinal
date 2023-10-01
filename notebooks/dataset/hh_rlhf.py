from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf")['train']


def make_prompt(example):
    return {"prompt": example["chosen"]}


dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(5000))
dataset = dataset.map(make_prompt)
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/hh_rlhf")

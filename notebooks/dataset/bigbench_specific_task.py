from datasets import load_dataset

dataset = load_dataset("tasksource/bigbench", "bbq_lite_json")['train']

dataset = dataset.shuffle(seed=42)
options = ['A', 'B', 'C']


def make_prompt(example):
    prompt = f"""{example['inputs']}{example['targets'][0]}"""
    return {"prompt": prompt}


dataset = dataset.map(make_prompt)
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/bbq_lite_json")
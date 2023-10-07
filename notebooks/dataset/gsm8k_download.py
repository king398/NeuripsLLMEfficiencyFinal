import datasets

dataset = datasets.load_dataset("gsm8k", "main")['train']


def make_prompt(example):
    prompt = f"""Q:{example['question']}
    A:{example['answer']}"""
    return {"prompt": prompt}


dataset = dataset.map(make_prompt)
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/gsm8k")

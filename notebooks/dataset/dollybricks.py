from datasets import load_dataset

dataset = load_dataset("databricks/databricks-dolly-15k")['train']


def make_prompt(example):
    if example['context'] != "":
        prompt = f"""{example['instruction']}
Input:{example['context']}
Response:{example['response']}"""
    else:
        prompt = f"""{example['instruction']}
Response:{example['response']}"""
    return {"prompt": prompt}


dataset = dataset.map(make_prompt)
print(dataset[1]['prompt'])
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/dollybricks")

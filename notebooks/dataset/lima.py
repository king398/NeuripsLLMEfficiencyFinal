import datasets

dataset = datasets.load_dataset("GAIR/lima", "plain_text")['train']


def make_prompt(example):
    prompt = f"""Input:{example['conversations'][0]}
Response:{example['conversations'][1]}"""
    return {"prompt": prompt}
dataset = dataset.map(make_prompt)
print(dataset[1]['prompt'])
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/lima")

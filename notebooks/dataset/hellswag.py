import datasets

dataset = datasets.load_dataset("Rowan/hellaswag")['train']
print(dataset[0]['endings'])


def make_prompt(example):
    prompt = f"Activity : {example['activity_label']} \nSentence:{example['ctx']} {example['endings'][int(example['label'])]} "
    return {"prompt": prompt}


dataset = dataset.map(make_prompt)
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(4000))
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/hellaswag")

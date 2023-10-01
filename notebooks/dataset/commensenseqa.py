import datasets

dataset = datasets.load_dataset("commonsense_qa")['train']
print(dataset[0])


def make_prompt(example):
    prompt = f"""Question: {example['question']}
    A: {example['choices']['text'][0]}
    B: {example['choices']['text'][1]}
    C: {example['choices']['text'][2]}
    D: {example['choices']['text'][3]}
    E: {example['choices']['text'][4]}
    Answer: {example['answerKey']}
    """
    return {"prompt": prompt}


dataset = dataset.map(make_prompt)
dataset = dataset.shuffle(seed=42)
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/commonsense_qa")

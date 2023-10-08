import datasets

dataset = datasets.load_dataset("Rowan/hellaswag")['validation']
print(dataset[0]['endings'])
answers = ['A', 'B', 'C', 'D']


def make_prompt(example):
    prompt = f"""Sentence:{example['ctx']}
Ending A: {example['endings'][0]}
Ending B: {example['endings'][1]}
Ending C: {example['endings'][2]}
Ending D: {example['endings'][3]}
Answer: {answers[int(example['label'])]}. {example['endings'][int(example['label'])]}"""
    return {"prompt": prompt}


dataset = dataset.map(make_prompt)
dataset = dataset.shuffle(seed=42)
print(dataset[0])
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/hellaswag")

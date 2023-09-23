import datasets

dataset = datasets.load_dataset("openbookqa", "main")['train']
def make_prompt(example):
    example['choices'] = example['choices']['text']
    prompt = f"""Question: {example['question_stem']} 
    A. {example['choices'][0]}
    B. {example['choices'][1]}
    C. {example['choices'][2]}
    D. {example['choices'][3]}
    Answer: {example['answerKey']}
    """
    return {"prompt": prompt}
dataset  = dataset.map(make_prompt)
dataset.save_to_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/openbookqa")


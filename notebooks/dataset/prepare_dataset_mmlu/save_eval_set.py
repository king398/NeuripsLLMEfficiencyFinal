from datasets import load_dataset

dataset = load_dataset("cais/mmlu", 'all')['test']

options_list = ["A", "B", "C", "D"]


def make_prompt(example):
    prompt = f"""Question: {example['question']} 
    A. {example['choices'][0]}
    B. {example['choices'][1]}
    C. {example['choices'][2]}
    D. {example['choices'][3]}
    Answer: {options_list[example['answer']]}
    """
    return {"prompt": prompt}


dataset = dataset.map(make_prompt)
print(dataset['answer'][0])
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/mmlu_test")

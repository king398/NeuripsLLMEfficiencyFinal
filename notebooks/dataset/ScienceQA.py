from datasets import load_dataset

def make_prompt(example):
    options = ["A", "B", "C", "D"]
    choices = '\n'.join([f"{opt}: {choice}" for opt, choice in zip(options, example['choices'])])
    answer = options[int(example['answer'])] if int(example['answer']) < len(example['choices']) else "Invalid"

    return {"prompt": f"Question: {example['question']}\n{choices}\nAnswer: {answer}"}

dataset = load_dataset("tasksource/ScienceQA_text_only")['train'].map(make_prompt)
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/ScienceQA")

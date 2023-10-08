from datasets import load_dataset
import random

dataset_sciq = load_dataset("sciq")['train']


def make_prompt(example):
    options = [example['distractor1'], example['distractor2'], example['distractor3'], example['correct_answer']]
    random.shuffle(options)
    prompt = f"""Question: {example['question']}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer: {example['correct_answer']}    
Support: {example['support']}"""
    return {"prompt": prompt}


dataset_sciq = dataset_sciq.map(make_prompt)
dataset_sciq.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/sciq")

import datasets
import pandas as pd
from tqdm import tqdm
import re
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf", use_fast=True)
openbooqa_main = datasets.load_dataset('openbookqa', 'main', split='train')
bigbench = datasets.load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/bigbench_train_small")
bigbench = bigbench.shuffle(seed=42)
cnn_dailymail = datasets.load_dataset("cnn_dailymail", "2.0.0")['train']
cnn_dailymail = cnn_dailymail.shuffle(seed=42)
cnn_dailymail = cnn_dailymail.select(range(int(len(cnn_dailymail) * 0.05)))
gsm8k = datasets.load_dataset('gsm8k', 'main')['train']
prompts = []


def convert_scores_to_letter_bigbench(multiple_choice_scores):
    # Find the index of the highest score
    max_index = multiple_choice_scores.index(max(multiple_choice_scores))

    # Convert the index to a letter from "A" to "E"
    answer_letter = chr(65 + max_index)

    return answer_letter


# Function to create prompt for BigBench
def create_prompt_bigbench(entry):
    inputs = entry['inputs']
    if inputs.endswith(" A:"):
        inputs = inputs[:-3]

    if not entry['multiple_choice_targets']:
        prompt = f"""{inputs}
Answer: {entry['targets'][0]}"""
    else:
        choices_str = "\n".join(
            [f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(entry['multiple_choice_targets'])]
        )
        prompt = f"""{inputs}
{choices_str}
Answer: {convert_scores_to_letter_bigbench(entry['multiple_choice_scores'])}"""
    prompt = re.sub(r'\nA:', '', prompt)

    return prompt


for i in openbooqa_main:
    prompt = f"""Question: {i['question_stem']}
A. {i['choices']['text'][0]}
B. {i['choices']['text'][1]}
C. {i['choices']['text'][2]}
D. {i['choices']['text'][3]}
Answer:{i['answerKey']}"""
    prompts.append(prompt)
for i in tqdm(bigbench):
    break
    prompt = create_prompt_bigbench(i)
    prompts.append(prompt)
for i in tqdm(cnn_dailymail):

    prompt = f"""Document: {i['article']}
Summary: {i['highlights']}"""
    prompts.append(prompt)
for i in tqdm(gsm8k):

    prompt = f"""question: {i['question']}
answer: {i['answer']}"""

data = {'prompts': prompts}
dataset = datasets.Dataset.from_dict(data)
import numpy as np


def add_token_count(batch):
    # Tokenize the batch of prompts and count the number of tokens for each
    token_counts = [len(t) for t in tokenizer(batch['prompts'])['input_ids']]

    return {'token_count': np.array(token_counts)}


# Once the dataset is updated, you can calculate the maximum token count

dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/training_prompts")

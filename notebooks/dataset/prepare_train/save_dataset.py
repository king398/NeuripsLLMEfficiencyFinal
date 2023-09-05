import datasets
import pandas as pd
from tqdm import tqdm
import re
openbooqa_main = datasets.load_dataset('openbookqa', 'main', split='train')
openbooqa_additional = datasets.load_dataset('openbookqa', 'additional', split='train')
bigbench = datasets.load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/bigbench_train")
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

for i in openbooqa_additional:
    prompt = f"""Question: {i['question_stem']}
A. {i['choices']['text'][0]}
B. {i['choices']['text'][1]}
C. {i['choices']['text'][2]}
D. {i['choices']['text'][3]}
Answer:{i['answerKey']}"""
    prompts.append(prompt)

for i in tqdm(bigbench):
    prompt = create_prompt_bigbench(i)
    prompts.append(prompt)

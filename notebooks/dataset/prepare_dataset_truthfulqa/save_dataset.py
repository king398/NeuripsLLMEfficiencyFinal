from datasets import load_dataset, concatenate_datasets
import random

dataset_generation = load_dataset("truthful_qa", "generation")['validation']

dataset_mc = load_dataset("truthful_qa", "multiple_choice")['validation']


def make_prompt_generation(example):
    answer_list = example['incorrect_answers'] + [example['best_answer']]
    random.shuffle(answer_list)
    best_answer_index = answer_list.index(example['best_answer'])
    question_format = f"Question:{example['question']}\n"
    formatted_answers = '+\n'.join(
        [f"{chr(65 + i)}: {answer}" for i, answer in enumerate(answer_list)])

    answer_format = f"\nAnswer:{chr(65 + best_answer_index)}"
    prompt = question_format + formatted_answers + answer_format
    return {"prompt": prompt}




def make_prompt_mc(example):
    choices = example['mc1_targets']['choices']
    labels = example['mc1_targets']['labels']
    combined = list(zip(choices, labels))
    random.shuffle(combined)
    choices, labels = zip(*combined)

    best_answer_index = labels.index(1)
    question_format = f"Question: {example['question']}\n"
    formatted_answers = '\n'.join([f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)])
    answer_format = f"\nAnswer: {chr(65 + best_answer_index)}"

    prompt = question_format + formatted_answers + answer_format
    return {"prompt": prompt}


dataset_generation = dataset_generation.map(make_prompt_generation)
dataset_mc = dataset_mc.map(make_prompt_mc)
dataset = concatenate_datasets([dataset_mc, dataset_generation])
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/truthful_qa")

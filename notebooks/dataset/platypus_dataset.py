from datasets import load_dataset, load_from_disk,Dataset

dataset = load_dataset("garage-bAInd/Open-Platypus")['train']
dataset = dataset.filter(lambda example: example['data_source'] != "airoboros")
dataset_cnn  = load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/cnn_dailymail_2_0")


def make_prompt(example):
    if example['input'] != "":
        prompt = f"""{example['input']}
Instruction:{example['instruction']}
Output:{example['output']}"""
    else:
        prompt = f"""Instruction:{example['instruction']}
Output:{example['output']}"""
    return {"prompt": prompt}


dataset = dataset.map(make_prompt)
prompts = dataset['prompt'] + dataset_cnn['prompt']
data = {"prompt": prompts}
dataset = Dataset.from_dict(data)
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/platypus")

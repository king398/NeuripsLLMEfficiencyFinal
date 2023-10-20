from datasets import load_dataset, load_from_disk, Dataset

dataset = load_dataset("garage-bAInd/Open-Platypus")['train']
exclude = ['airoboros',"MATH/PRM-800K","leetcode_ne"]
dataset = dataset.filter(lambda example: example['data_source'] not in exclude)


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
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/platypus")

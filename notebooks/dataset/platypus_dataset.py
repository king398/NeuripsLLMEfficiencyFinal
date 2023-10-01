from datasets import load_dataset

dataset = load_dataset("garage-bAInd/Open-Platypus")['train']
dataset = dataset.filter(lambda example: example['data_source'] != "airoboros")


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

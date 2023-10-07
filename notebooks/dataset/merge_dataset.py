from datasets import load_dataset, load_from_disk, Dataset

dataset_cnn = load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/cnn_dailymail_2_0")

dataset_openbookqa = load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/openbookqa")
dataset_sciq = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/sciq")


def make_prompt(example):
    if example['input'] != "":
        prompt = f"""{example['input']}
Instruction:{example['instruction']}
Output:{example['output']}"""
    else:
        prompt = f"""Instruction:{example['instruction']}
Output:{example['output']}"""
    return {"prompt": prompt}


prompts = dataset_cnn['prompt'] + dataset_openbookqa['prompt'] + dataset_sciq['prompt']
data = {"prompt": prompts}
dataset = Dataset.from_dict(data)
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/cnn-openbookqa-sciq")
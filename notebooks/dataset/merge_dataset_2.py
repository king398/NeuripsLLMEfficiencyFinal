from datasets import load_dataset, load_from_disk, Dataset

dataset_cnn = load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/cnn_dailymail_2_0")
dataset_dollybricks = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/dollybricks")
dataset_platypus = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/platypus")
commonsense_qa = load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/commonsense_qa")
prompts = dataset_cnn['prompt'] + dataset_dollybricks['prompt'] + dataset_platypus['prompt'] + commonsense_qa['prompt']
data = {"prompt": prompts}
dataset = Dataset.from_dict(data)
dataset.save_to_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/cnn_dollybricks")

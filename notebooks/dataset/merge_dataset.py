from datasets import load_dataset, load_from_disk, Dataset

dataset_cnn = load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/cnn_dailymail_2_0")

dataset_openbookqa = load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/openbookqa")
dataset_sciq = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/sciq")

dataset_platypus = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/platypus")
dataset_dollybricks = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/dollybricks")
prompts = dataset_cnn['prompt'] + dataset_platypus['prompt'] +  dataset_dollybricks['prompt']
data = {"prompt": prompts}
dataset = Dataset.from_dict(data)
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/cnn-platypus-dollybricks")

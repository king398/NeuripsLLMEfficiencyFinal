from datasets import load_dataset, load_from_disk, Dataset

dataset_cnn = load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/cnn_dailymail_2_0")
dataset_dollybricks = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/dollybricks")
dataset_platypus = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/platypus")
dataset_bbq = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/bbq_lite_json")
prompts = dataset_cnn['prompt'] + dataset_dollybricks['prompt'] + dataset_platypus['prompt'] + dataset_bbq['prompt']
data = {"prompt": prompts}
dataset = Dataset.from_dict(data)
dataset.save_to_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/cnn_2_0_0_dollybricks_platypus_bbq")

from datasets import load_dataset, load_from_disk, Dataset

dataset_cnn = load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/cnn_dailymail_2_0")

dataset_openbookqa = load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/openbookqa")
dataset_sciq = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/sciq")
dataset_ScienceQA = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/ScienceQA")
dataset_lima = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/lima")
dataset_commonsense = load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/commonsense_qa")
prompts = dataset_cnn['prompt'] + dataset_sciq['prompt'] + dataset_openbookqa['prompt'] + dataset_ScienceQA['prompt'] + \
          dataset_lima['prompt'] + dataset_commonsense['prompt']
data = {"prompt": prompts}
dataset = Dataset.from_dict(data)
dataset.save_to_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/merged_datasets/openbookqa_scienceqa_cnn_sciq_lima_commonsense")

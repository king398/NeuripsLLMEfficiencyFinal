from datasets import load_dataset, concatenate_datasets

dataset_generation = load_dataset("truthful_qa", "generation")['validation']
dataset_generation.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/truthful_qa_generation.hf")

dataset_mc = load_dataset("truthful_qa", "multiple_choice")['validation']
dataset_mc.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/truthful_qa_mc.hf")

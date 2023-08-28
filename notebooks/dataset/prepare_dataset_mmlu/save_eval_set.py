from datasets import load_dataset

dataset = load_dataset("cais/mmlu",'all')
dataset.save_to_disk("mmlu_test.hf")

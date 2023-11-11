from datasets import load_dataset, load_from_disk, Dataset
dataset = load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/cnn_2_0_0_dollybricks_platypus_bbq")
dataset.push_to_hub("Mithilss/cnn_dollybricks_platypus_bbq_2_0",token="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")
from datasets import Dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import re
import ast
import datasets
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = datasets.load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/mmlu_test.hf")['test']
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": 0})
results = {"mmlu": []}
for i in tqdm(dataset, desc="Processing mmlu"):

    prompt = f"""Question: {i['question']}
    A. {i['choices'][0]}
    B. {i['choices'][1]}
    C. {i['choices'][2]}
    D. {i['choices'][3]}
    Answer:"""
    options = ['A', 'B', 'C', 'D']
    input_ids = tokenizer(prompt, return_tensors='pt')
    input_ids = {k: v.to(model.device) for k, v in input_ids.items()}

    with torch.no_grad():
        output = model.generate(**input_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    try:
        answer_llm = re.search(r"Answer:\s*([A-E])", generated_text, re.I).group(1).lower()
    except:
        answer_llm = None
    truth = options[i['answer']].lower()
    results["mmlu"].append(answer_llm == truth)
for dataset_name, dataset_results in results.items():
    try:
        accuracy = (dataset_results.count(True) / len(dataset_results)) * 100
    except ZeroDivisionError:
        accuracy = 0
    print(f"Accuracy for {dataset_name}: {accuracy:.2f}% With Model {model_name}")

# Print overall accuracy
all_results = [item for sublist in results.values() for item in sublist]

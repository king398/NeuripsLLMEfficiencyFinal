from datasets import Dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import re
import datasets
import json
import peft

model_path = "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/models/codellama-2-7b-baseline-lr-1e-4/checkpoint-4828"
model_name = "codellama/CodeLlama-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset_mmlu = datasets.load_from_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/mmlu_test.hf")['test']
dataset_bbq = pd.read_csv("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/bbq_test.csv")
dataset_bbq = Dataset.from_pandas(dataset_bbq)
dataset_truthful_qa_generation = datasets.load_from_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/truthful_qa_generation.hf")
dataset_truthful_qa_mc = datasets.load_dataset("truthful_qa", "multiple_choice")['validation']
from transformers import BitsAndBytesConfig



model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": 0},
                                             load_in_8bit=True
                                             )
#model = peft.PeftModel.from_pretrained(model, model_path)
results = {"mmlu": [], "bbq": [], 'truthful_qa_mc': [], 'truthful_qa_generation': []}
for i in tqdm(dataset_truthful_qa_mc, desc="Processing truthful_qa_mc"):

    prompt = f"Question: {i['question']}\n"
    prompt += f"A. {i['mc1_targets']['choices'][0]}\n"
    for idx, answer in enumerate(i['mc1_targets']['choices']):
        alphabet = chr(66 + idx + 1)  # 66 is ASCII for 'B'
        prompt += f"{alphabet}. {answer}\n"
    prompt += "Answer:"
    input_ids = tokenizer(prompt, return_tensors='pt')
    input_ids = {k: v.to(model.device) for k, v in input_ids.items()}
    with torch.no_grad():
        output = model.generate(**input_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    try:
        answer_llm = re.search(r"Answer:\s*([A-X])", generated_text, re.I).group(1).lower()
    except:
        answer_llm = None
    truth = 'a'
    results["truthful_qa_mc"].append(answer_llm == truth)

for i in tqdm(dataset_truthful_qa_generation, desc="Processing truthful_qa_generation"):

    prompt = f"Question: {i['question']}\n"

    # Add the correct (or best) answer first and label it "A"
    prompt += f"A. {i['best_answer']}\n"

    # Loop through the list of incorrect answers and append them after the correct answer
    for idx, answer in enumerate(i['incorrect_answers']):
        alphabet = chr(66 + idx)  # 66 is ASCII for 'B'
        prompt += f"{alphabet}. {answer}\n"
    prompt += "Answer:"
    input_ids = tokenizer(prompt, return_tensors='pt')
    input_ids = {k: v.to(model.device) for k, v in input_ids.items()}
    with torch.no_grad():
        output = model.generate(**input_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    try:
        answer_llm = re.search(r"Answer:\s*([A-X])", generated_text, re.I).group(1).lower()
    except:
        answer_llm = None
    truth = 'a'
    results["truthful_qa_generation"].append(answer_llm == truth)

for i in tqdm(dataset_bbq, desc="Processing bbq"):

    prompt = f"""Context: {i['context']}
    Question: {i['question']}
    A. {i['ans0']}
    B. {i['ans1']}
    C. {i['ans2']}
    Answer:"""
    options = ['A', 'B', 'C']
    input_ids = tokenizer(prompt, return_tensors='pt')
    input_ids = {k: v.to(model.device) for k, v in input_ids.items()}
    with torch.no_grad():
        output = model.generate(**input_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    try:
        answer_llm = re.search(r"Answer:\s*([A-C])", generated_text, re.I).group(1).lower()
    except:
        answer_llm = None
    truth = options[i['label']].lower()
    results["bbq"].append(answer_llm == truth)
for i in tqdm(dataset_mmlu, desc="Processing mmlu"):

    prompt = f"""
    Question: {i['question']}
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
results_dict = {}
for dataset_name, dataset_results in results.items():
    try:
        accuracy = (dataset_results.count(True) / len(dataset_results)) * 100
    except ZeroDivisionError:
        accuracy = 0
    print(f"Accuracy for {dataset_name}: {accuracy:.2f}% With Model {model_name}")
    results_dict[dataset_name] = accuracy

# Print overall accuracy
all_results = [item for sublist in results.values() for item in sublist]
with open(f"/home/mithil/PycharmProjects/NeuripsLLMEfficiency/eval/{model_path.split('/')[-2]}.json", "w") as f:
    json.dump(results_dict, f)

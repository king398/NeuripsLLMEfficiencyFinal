from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_name = "mosaicml/mpt-30b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto",
                                             trust_remote_code=True,load_in_8bit=True)
information = """An alloy is a mixture of chemical elements of which at least one is a metal. Unlike chemical compounds with metallic bases, an alloy will retain all the properties of a metal in the resulting material, such as electrical conductivity, ductility, opacity, and luster, but may have properties that differ from those of the pure metals, such as increased strength or hardness. In some cases, an alloy may reduce the overall cost of the material while preserving important properties. In other cases, the mixture imparts synergistic properties to the constituent metal elements such as corrosion resistance or mechanical strength"""
prompt = f"""Given the information provided below, create a multiple choice question with four options, ensuring only one of them is the correct answer. The question should follow this format:
Question: [Your generated question here]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
Answer: [Correct answer choice, e.g., A, B, C, or D]
Information: {information}
Question:"""
input_ids = tokenizer(prompt, return_tensors='pt')
input_ids = {k: v.to(model.device) for k, v in input_ids.items()}
with torch.no_grad():
    start = time.time()
    output = model.generate(**input_ids, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id, do_sample=True,
                            top_k=5)
    end = time.time()
print(f"Time taken: {end - start}")
per_token_time = (end - start) / 100
tokens_per_sec = 1 / per_token_time
print(f"Per token time: {per_token_time} seconds , Tokens per second: {tokens_per_sec} ")

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

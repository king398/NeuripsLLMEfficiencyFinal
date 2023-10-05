from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "tiiuae/falcon-40b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto",
                                             use_flash_attn=False, trust_remote_code=True)
prompt = f"""Given the information provided below, create a multiple choice question with four options, ensuring only one of them is the correct answer. The question should follow this format:
Question: [Your generated question here]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
Answer: [Correct answer choice, e.g., A, B, C, or D]
Information:
"International Atomic Time (TAI, from the French name ) is a high-precision atomic coordinate time standard based on the notional passage of proper time on Earth's geoid. It is a continuous scale of time, without leap seconds. It is the principal realisation of Terrestrial Time (with a fixed offset of epoch). It is also the basis for Coordinated Universal Time (UTC), which is used for civil timekeeping all over the Earth's surface. UTC deviates from TAI by a number of whole seconds. , when another leap second was put into effect, UTC is currently exactly 37 seconds behind TAI. The 37 seconds result from the initial difference of 10 seconds at the start of 1972, plus 27 leap seconds in UTC since 1972."
Question:"""
input_ids = tokenizer(prompt, return_tensors='pt')
input_ids = {k: v.to(model.device) for k, v in input_ids.items()}
with torch.no_grad():
    output = model.generate(**input_ids, max_new_tokens=125, pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

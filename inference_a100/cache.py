import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure the logging module
model_name = "Qwen/Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True, use_flash_attn=True,
                                             ).eval()

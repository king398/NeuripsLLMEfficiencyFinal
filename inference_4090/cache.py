import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure the logging module
model_name = "Qwen/Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True
                                             ).eval()
model = PeftModel.from_pretrained(model, "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/models/Qwen/Qwen-14B-1-epoch-cnn-openbookqa-sciq-lima/checkpoint-2041")
model = model.merge_and_unload()
model.save_pretrained("qwen-14b-finetune")

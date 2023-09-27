import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login(token=os.environ["HUGGINGFACE_TOKEN"])

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
model_name = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map={"": 0},
                                             load_in_8bit=True)

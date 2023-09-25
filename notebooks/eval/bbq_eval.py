from fastapi import FastAPI
import bitsandbytes as bnb

import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import peft
from torch.cuda.amp import autocast

# login(token=os.environ["HUGGINGFACE_TOKEN"])

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map={"": 0}, )
model = peft.PeftModel.from_pretrained(model,
                                       "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/models/Llama-2-7b-hf-2-epoch/checkpoint-5022")

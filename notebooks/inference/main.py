from fastapi import FastAPI

import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)
model_name = "Mithilss/Llama-2-7b-hf"
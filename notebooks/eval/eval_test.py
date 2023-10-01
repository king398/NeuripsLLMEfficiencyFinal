import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)
model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0},
                                             trust_remote_code=True,torch_dtype=torch.bfloat16 ).eval()
prompt = f"ok"
encoded = tokenizer(prompt, return_tensors="pt")
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
encoded = {k: v.to("cuda") for k, v in encoded.items()}
import time
with torch.no_grad():
    outputs = model.generate(
        **encoded,
        max_new_tokens=25,
    )
start = time.time()
with torch.no_grad():

        for i in range(1000):
            outputs = model(
                **encoded,
            )
end = time.time()
print(end - start)

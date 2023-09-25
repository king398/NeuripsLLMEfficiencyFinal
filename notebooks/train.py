import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers.trainer_callback import TrainerCallback
import torch
import os
import pandas as pd



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class CFG:
    WANDB_PROJECT = 'NeuripsLLMEfficiency'
    CUDA_VISIBLE_DEVICES = "0"
    PRETRAINED_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    DATASET_PATH = "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/all_prompts"
    output_dir = "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/models/Llama-2-7b-hf-2-epoch"
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        num_train_epochs=2,
        bf16_full_eval=True,
        bf16=True,
        output_dir=output_dir,
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        save_strategy="epoch",
        overwrite_output_dir=True,
        save_total_limit=2,
        learning_rate=1e-4,
        optim="adamw_torch",
        seed=42,
        tf32=True,
        logging_steps=1,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,

    )


os.environ['WANDB_PROJECT'] = CFG.WANDB_PROJECT
os.environ["CUDA_VISIBLE_DEVICES"] = CFG.CUDA_VISIBLE_DEVICES
tokenizer = AutoTokenizer.from_pretrained(CFG.PRETRAINED_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CFG.PRETRAINED_MODEL_NAME,
    torch_dtype=torch.bfloat16, device_map="auto")
# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()
model.config.use_cache = False
modules = find_all_linear_names(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules)
dataset = datasets.load_from_disk(CFG.DATASET_PATH)


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    max_seq_length=1536,
    args=CFG.training_args,
    tokenizer=tokenizer,
    dataset_text_field="prompt",
    peft_config=peft_config,
    callbacks=[PeftSavingCallback()],


)

trainer.train()

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
    WANDB_PROJECT = 'NeuripsLLMEfficiency2'
    PRETRAINED_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    DATASET_PATH = "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/cnn-openbookqa-sciq-hellaswag"
    output_dir = "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/models/mistralai/Mistral-7B-v0.1-1-epoch-cnn-openbookqa-sciq-hellaswag-small-dataset"
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        num_train_epochs=3,
        bf16_full_eval=True,
        bf16=True,
        output_dir=output_dir,
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        save_strategy="epoch",
        overwrite_output_dir=True,
        save_total_limit=3,
        learning_rate=1e-5,
        optim="adamw_torch",
        seed=42,
        tf32=True,
        logging_steps=1,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0,
        save_safetensors=True,

    )


os.environ['WANDB_PROJECT'] = CFG.WANDB_PROJECT
tokenizer = AutoTokenizer.from_pretrained(CFG.PRETRAINED_MODEL_NAME, trust_remote_code=True, truncation=True,
                                          padding=False, max_length=2048)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(CFG.PRETRAINED_MODEL_NAME, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)

model.gradient_checkpointing_enable()
model.config.use_cache = False
modules = find_all_linear_names(model)
print(modules)
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
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
    max_seq_length=2048,
    args=CFG.training_args,
    tokenizer=tokenizer,
    dataset_text_field="prompt",
    peft_config=peft_config,
    callbacks=[PeftSavingCallback()],

)

trainer.train()
trainer.save_model()

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers.trainer_callback import TrainerCallback
import torch
import os
import pandas as pd


class CFG:
    WANDB_PROJECT = 'NeuripsLLM-'
    CUDA_VISIBLE_DEVICES = "0"
    PRETRAINED_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    DATASET_PATH = "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/training_prompts"
    output_dir = "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/models/Llama-2-7b-baseline-small-finetune"
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        num_train_epochs=1,
        bf16_full_eval=True,
        bf16=True,
        output_dir=output_dir,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        save_strategy="epoch",
        overwrite_output_dir=True,
        save_total_limit=1,
        learning_rate=1e-4,
        optim="adamw_hf",
        seed=42,
        tf32=True,
        logging_steps=1,

    )


os.environ['WANDB_PROJECT'] = CFG.WANDB_PROJECT
os.environ["CUDA_VISIBLE_DEVICES"] = CFG.CUDA_VISIBLE_DEVICES
tokenizer = AutoTokenizer.from_pretrained(CFG.PRETRAINED_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CFG.PRETRAINED_MODEL_NAME,
    torch_dtype=torch.float16, device_map="auto")
# model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
model.gradient_checkpointing_enable()
model.config.use_cache = False
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM", )
# target_modules=modules,
dataset = datasets.load_from_disk(CFG.DATASET_PATH)
print(len(dataset))


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    max_seq_length=1024,
    args=CFG.training_args,
    tokenizer=tokenizer,
    dataset_text_field="prompts",
    peft_config=peft_config,
    callbacks=[PeftSavingCallback()],
)

trainer.train()

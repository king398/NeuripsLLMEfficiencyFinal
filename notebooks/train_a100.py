import os
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, TrainerCallback, BitsAndBytesConfig
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.cuda.amp import autocast


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
    max_length = 2200
    WANDB_PROJECT = 'NeuripsLLMEfficiency2'
    PRETRAINED_MODEL_NAME = "Qwen/Qwen-14B"
    DATASET_PATH = "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/cnn_2_0_0_dollybricks_platypus_bbq"
    output_dir = "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/models/Qwen/Qwen-14B-1-cnn_dollybricks_platypus_bbq_rank_16"
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        num_train_epochs=1,
        fp16_full_eval=True,
        fp16=True,
        output_dir=output_dir,
        gradient_checkpointing=True,
        gradient_accumulation_steps=16,
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
                                          padding=False, max_length=CFG.max_length, )
tokenizer.padding_side = "right"
tokenizer.pad_token = "<|endoftext|>"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
    ,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(CFG.PRETRAINED_MODEL_NAME, torch_dtype=torch.float16,
                                             trust_remote_code=True, quantization_config=nf4_config, device_map="auto",
                                             use_flash_attn=False)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.config.use_cache = False

modules = find_all_linear_names(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
dataset = datasets.load_from_disk(CFG.DATASET_PATH)


# Tokenize the dataset
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["prompt"], truncation=True, padding="longest", max_length=CFG.max_length)
    return tokenized_inputs


# Tokenize the dataset.
tokenized_datasets = dataset.map(tokenize_function, batched=False, num_proc=16)
list_of_lens = []
for i in range(len(tokenized_datasets)):
    list_of_lens.append(len(tokenized_datasets[i]['input_ids']))
max_len_index = list_of_lens.index(max(list_of_lens))
print(tokenized_datasets[max_len_index]['prompt'])
print(len(tokenizer(tokenized_datasets[max_len_index]['prompt'])['input_ids']))
print(list_of_lens.count(CFG.max_length))
# plot histogram of lengths


# Now, we create a function to sort the tokenized dataset based on sequence length.


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


# Use the Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=CFG.training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),  # for causal LM
    callbacks=[PeftSavingCallback()],
)

with autocast(dtype=torch.float16):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        trainer.train()
trainer.save_model(CFG.output_dir)

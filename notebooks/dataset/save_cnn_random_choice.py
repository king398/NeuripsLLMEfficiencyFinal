import random
from datasets import load_dataset


# Define a function to format the examples.
def format_example(example):
    return f"###\nArticle: {example['article']}\n\nSummarize the above article in 3 sentences.\n{example['highlights']}\n"


# Function to pick random examples from a dataset and format them as summaries.
def pick_random_samples(dataset, sample_size=1):
    random_indices = random.sample(range(len(dataset)), sample_size)
    return [format_example(dataset[idx]) for idx in random_indices]


# Function to create a two-shot prompt using examples from a different subset.
def make_prompt(example, index, all_indices, random_sample_dataset):
    # Exclude the current index from available indices
    try:
        available_indices = [idx for idx in all_indices if idx != index]
        example_prompts = pick_random_samples(random_sample_dataset, 1)

        # Prepare the current example
        current_example = format_example(example)

        # Combine the examples and the current prompt
        prompt = "".join(example_prompts) + current_example
    except:
        prompt = "###\nArticle: \n\nSummarize the above article in 3 sentences.\n"
    return {"prompt": prompt}


# Load and prepare the dataset.
full_dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0")['train']
full_dataset = full_dataset.shuffle(seed=42)
dataset_size = 30000
prompt_creation_dataset = full_dataset.select(range(dataset_size))
random_sample_dataset = full_dataset.select(range(dataset_size, len(full_dataset)))

# Indices for the random sampling, excluding the prompt creation subset
all_indices = list(range(dataset_size, len(full_dataset)))

# Create prompts using the 'map' function, with additional arguments for 'make_prompt'
dataset_with_prompts = prompt_creation_dataset.map(
    make_prompt,
    with_indices=True,  # Important to receive the 'index' in 'make_prompt'
    fn_kwargs={'all_indices': all_indices, 'random_sample_dataset': random_sample_dataset},
)
dataset_with_prompts = dataset_with_prompts.remove_columns(["article","highlights","id"])
# Save the modified dataset with prompts
dataset_with_prompts.save_to_disk(
    "/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/cnn_dailymail_2_0")

import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset

# Load subsets and concatenate them
subsets = ['Disability_status', 'Age', 'Gender_identity', 'Nationality', 'Physical_appearance', 'Race_ethnicity',
           'Race_x_gender', 'Religion', 'SES', 'Sexual_orientation']
dataset = concatenate_datasets([load_dataset("heegyu/bbq", subset)['test'] for subset in subsets])

# Shuffle and split the dataset
dataset = dataset.shuffle(seed=42)
choice_int = ['A', 'B', 'C']


def make_prompt(example):
    prompt = f"""Context: {example['context']}
    Question:{example['question']}
    A.{example['ans0']}
    B.{example['ans1']}
    C.{example['ans2']}
    Answer: {choice_int[example['label']]}
    """
    return {"prompt": prompt}


dataset = dataset.map(make_prompt)
df = pd.DataFrame(dataset)
df = df.drop_duplicates(subset=['ans2'], keep='first')

deduplicated_dataset = Dataset.from_pandas(df)

deduplicated_dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/filtered_datasets/BBQ")

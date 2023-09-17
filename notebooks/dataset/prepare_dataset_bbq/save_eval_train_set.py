from datasets import load_dataset, concatenate_datasets, load_metric, DatasetDict
import pandas as pd

# Load subsets and concatenate them
subsets = ['Disability_status', 'Age', 'Gender_identity', 'Nationality', 'Physical_appearance', 'Race_ethnicity', 'Race_x_gender', 'Religion', 'SES', 'Sexual_orientation']
dataset = concatenate_datasets([load_dataset("heegyu/bbq", subset)['test'] for subset in subsets])

# Shuffle and split the dataset
dataset = dataset.shuffle(seed=42)
print(dataset)
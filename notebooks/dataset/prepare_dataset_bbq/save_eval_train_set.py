from datasets import load_dataset, concatenate_datasets, load_metric, DatasetDict
import pandas as pd

# Load subsets and concatenate them
subsets = ['Disability_status', 'Age', 'Gender_identity', 'Nationality', 'Physical_appearance', 'Race_ethnicity', 'Race_x_gender', 'Religion', 'SES', 'Sexual_orientation']
dataset = concatenate_datasets([load_dataset("heegyu/bbq", subset)['test'] for subset in subsets])

# Shuffle and split the dataset
dataset = dataset.shuffle(seed=42)
train_test_split = dataset.train_test_split(test_size=0.2)

# Convert to pandas DataFrame
train_df = train_test_split['train'].to_pandas()
test_df = train_test_split['test'].to_pandas()

# Save as CSV files
train_df.to_csv("bbq_train.csv", index=False)
test_df.to_csv("bbq_test.csv", index=False)

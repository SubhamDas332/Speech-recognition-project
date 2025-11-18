import os
import pandas as pd
from datasets import Dataset, DatasetDict

def prepare_data(data_dir):
    """
    Loads train, dev, and test CSV files and creates a Hugging Face DatasetDict.
    The 'file' column is updated to absolute paths.
    """
    # CSV file paths
    train_csv = os.path.join(data_dir, 'train.csv')
    dev_csv = os.path.join(data_dir, 'dev.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    # Load CSVs into pandas DataFrames
    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)
    test_df = pd.read_csv(test_csv)

    # Prepend data_dir to 'file' column (only once)
    for df in [train_df, dev_df, test_df]:
        df['file'] = df['file'].apply(lambda x: os.path.join(data_dir, x))

    # Convert to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': dev_dataset,
        'test': test_dataset
    })

    return dataset_dict

if __name__ == '__main__':
    # Full path to the geo folder
    data_dir = '/work/courses/T/S/89/5150/general/data/geo_ASR_challenge_2025'
    prepared_dataset = prepare_data(data_dir)

    # Save to disk
    prepared_dataset.save_to_disk('prepared_dataset')

    print("Dataset prepared and saved to 'prepared_dataset'")
    print(prepared_dataset)
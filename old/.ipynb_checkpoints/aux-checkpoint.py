#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

class VocabularyBuilder:
    """
    Dynamically generates the character vocabulary (CHAR_MAP) 
    and inverse map (INV_CHAR_MAP) from combined transcript columns 
    of training and development CSV files.
    """
    def __init__(self, train_csv_path, val_csv_path):
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path

    def build_vocab(self):
        """
        Loads transcripts, normalizes them, and creates the character map.

        Returns:
            tuple: (CHAR_MAP, INV_CHAR_MAP, VOCAB_SIZE)
        """
        try:
            # Load data
            df_train = pd.read_csv(self.train_csv_path)
            df_val = pd.read_csv(self.val_csv_path)
        except FileNotFoundError as e:
            print("="*60)
            print("ERROR: CSV files not found!")
            print(f"Please check paths: {self.train_csv_path} and {self.val_csv_path}")
            print("="*60)
            raise e

        # 1. Combine and normalize transcripts
        all_transcripts = pd.concat([df_train['transcript'], df_val['transcript']])
        # Apply normalization: strip whitespace and convert to lowercase
        all_transcripts_normalized = all_transcripts.astype(str).str.strip().str.lower()

        # 2. Collect unique characters
        unique_chars = set()
        for transcript in all_transcripts_normalized:
            unique_chars.update(transcript)

        # 3. Sort characters for deterministic mapping
        # Space should be treated as a character but sorted specifically.
        char_list = sorted(list(unique_chars - set([' '])))
        if ' ' in unique_chars:
            char_list.insert(0, ' ')

        # 4. Generate the CHAR_MAP
        CHAR_MAP = {}
        # Assign IDs 0, 1, 2 to special tokens
        CHAR_MAP['<PAD>'] = 0
        CHAR_MAP['<SOS>'] = 1
        CHAR_MAP['<EOS>'] = 2

        # Assign IDs starting from 3 to the actual characters
        for i, char in enumerate(char_list):
            CHAR_MAP[char] = i + 3

        VOCAB_SIZE = len(CHAR_MAP)
        INV_CHAR_MAP = {v: k for k, v in CHAR_MAP.items()}
        
        return CHAR_MAP, INV_CHAR_MAP, VOCAB_SIZE

if __name__ == '__main__':
    # Example usage when running the file directly (requires 'geo/train.csv' and 'geo/dev.csv')
    try:
        builder = VocabularyBuilder('geo/train.csv', 'geo/dev.csv')
        char_map, _, vocab_size = builder.build_vocab()
        print(f"Test run successful. VOCAB_SIZE: {vocab_size}. Map keys: {list(char_map.keys())[:5]}...")
    except FileNotFoundError:
        pass


# In[ ]:





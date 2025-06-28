import pandas as pd, numpy as np
import os, torch,
from tensorflow.keras.preprocessing.text import Tokenizer as tk
from tensorflow.karas.preprocessing.sequence import pad_sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from datetime import datetime as dt

dataset_file = '/home/cory/code/Amik-Research-Testing/DATASETS/Text-Based Datasets/E-Commerce Product Dataset/product_names.csv'
output_path = '/home/cory/code/Amik-Research-Testing/ModelHub/OUTPUTS/Autocomplete/OUTPUTS'
now = dt.now().strftime('%y-%m-%d %H:%M:%S')
joined_output = os.path.join(output_path, f'Autocomplete-Keras_{now}')

data_load = pd.read_csv(dataset_file)
data = " ".join(data_load['Product'].astype(str))

# Tokenizing text
token = Tokenizer()
tk.fit_on_texts([data])
total_words = len(tk.word_index) + 1
token_list = tk.texts_to_sequences([text])[0] # Grabs the inner list of the list which is always at index 0








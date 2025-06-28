from sklearn.model_selection import train_test_split as tts
from datetime import datetime as dt
import tensorflow as tf, numpy as np, os, glob, pandas as pd
from transformers import BertTokenizer as bert

tflite_path = "**/mobilebert.tflite"
dataset_path = '**/VideoGameSales.csv'
output_path = 'home/cory/code/Anik-Research-Testing/ModelHub/OUTPUTS/Text-Classification/MobileBert'
now = dt.now().strftime('%y-%m-%d_%H-%M-%S')
joined_output = os.path.join(f'MobileBertOutput_{now}', output_path)
tokenizer = bert.from_pretrained("google/mobilebert-uncased")

df = pd.read_csv(dataset_path)

df = df[['Name', 'Genre']].dropna()
df['Genre'] = df['Genre'].astype(str)

# Columns or "Feature Sets"
df['Name']
df['Genre']

# Input data & Labels
input_data = df['Name']
label_data = df['Genre']

# Train/Test Split using train_test_split library
X_train, X_test, y_train, y_test = tts(input_data, label_data, test_size=0.2, random_state=42)

training_tokenization = tokenizer(list(X_train), return_tensors='np', padding='max_length', truncation=True, max_length=128)

testing_tokenization = tokenizer(list(X_test), return_tensors='np', padding='max_length', truncation=True, max_length=128)

# Tensorflow tflite load
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()



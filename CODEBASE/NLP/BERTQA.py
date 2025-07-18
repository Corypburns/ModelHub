import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
import time as t
import keyboard as k
from sklearn.model_selection import train_test_split as tts
import json

# === CONFIG ===
TRAIN_PATH = r"E:\Code\Python\ModelHub\DATASETS\NLP\Train\train-v2.0.json"
TEST_PATH = r"E:\Code\Python\ModelHub\DATASETS\NLP\Test\dev-v2.0.json"
MODEL_PATH = r"E:\Code\Python\ModelHub\MODELBASE\NLP\MobileBert.tflite"
LOG_DIR = r"E:\Code\Python\ModelHub\OUTPUTS\NLP"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_bertqa_{DATE_TIME}.csv"
OUTPUT_PATH = os.path.join(LOG_DIR, FILE_NAME)
HEADERS = (
    "Timestamp,Review,Mode,"
    "Pre_Lat_ms,Inf_Lat_ms,Post_Lat_ms,"
    "Pre_E_mJ,Inf_E_mJ,Post_E_mJ,"
    "Pre_Max_V,Pre_Mean_V,Pre_Max_C,Pre_Mean_C,"
    "Inf_Max_V,Inf_Mean_V,Inf_Max_C,Inf_Mean_C,"
    "Post_Max_V,Post_Mean_V,Post_Max_C,Post_Mean_C,"
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW\n"
)

# === LOADING DATASET ===
with open(TRAIN_PATH, 'r') as f:
    data = json.load(f)

sample_list = []
for article in data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            if len(qa['answers']) > 0:
                answer = qa['answers'][0]['text']
                answer_start = qa['answers'][0]['answer_start']
            else:
                # For unanswerable questions in SQuAD v2.0:
                answer = ""
                answer_start = -1
            sample_list.append((context, question, answer, answer_start))


train_samples, test_samples = tts(sample_list, test_size=0.1, random_state=42)

# === LOADING MODEL ===
start_load = t.time()
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
end_load = t.time()
load_total = (end_load - start_load) * 1000

start_allocate = t.time()
interpreter.allocate_tensors()
end_allocation = t.time()
allocation_total = (end_allocation - start_allocate) * 1000

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\nModel load time: {load_total:.4f} ms.")
print(f"Model allocation time: {allocation_total:.4f} ms.")

print(f"Train Samples: {len(train_samples)}")
print(f"Test Samples: {len(test_samples)}")

print(f"\nPress 'space' to continue...")
k.wait('space')
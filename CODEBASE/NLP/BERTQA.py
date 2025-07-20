import tensorflow as tf
import numpy as np
from transformers import BertTokenizer as BT
import json
from sklearn.model_selection import train_test_split as tts
import psutil
import os
import time
import csv
from datetime import datetime as dt

# TIMES LOGGED IN LOG FILE AREN'T ACCURATE, JUST USED THOSE FOR TESTING PURPOSES. WILL FIX IN LAB

# === CONFIG & PATHS ===
TRAIN_PATH = r"E:\Code\Python\ModelHub\DATASETS\NLP\Train\train-v2.0.json"
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

# Create output directory if needed
os.makedirs(LOG_DIR, exist_ok=True)

# Write CSV headers
with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
    f.write(HEADERS)

def append_csv_row(row_data):
    with open(OUTPUT_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

# Load tokenizer
tokenizer = BT.from_pretrained("google/mobilebert-uncased")

# Load SQuAD train data
with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
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
                answer = ""
                answer_start = -1
            sample_list.append((context, question, answer, answer_start))

# Split into train/test, keep test for inference
_, test_samples = tts(sample_list, test_size=0.1, random_state=42)

# Load model with timing
start_load = time.time()
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
end_load = time.time()

start_alloc = time.time()
interpreter.allocate_tensors()
end_alloc = time.time()

print(f"Model load time: {(end_load - start_load)*1000:.2f} ms")
print(f"Tensor allocation time: {(end_alloc - start_alloc)*1000:.2f} ms")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Process handle for memory
process = psutil.Process(os.getpid())

# Encode inputs
def encode(question, context, max_len=384):
    tokens = tokenizer.encode_plus(
        question,
        context,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    return (
        tokens['input_ids'].astype(np.int32),
        tokens['attention_mask'].astype(np.int32),
        tokens['token_type_ids'].astype(np.int32)
    )

# Run prediction with timing
def predict(input_ids, attention_mask, segment_ids):
    interpreter.set_tensor(input_details[0]['index'], input_ids)
    interpreter.set_tensor(input_details[1]['index'], attention_mask)
    interpreter.set_tensor(input_details[2]['index'], segment_ids)
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000
    start_logits = interpreter.get_tensor(output_details[0]['index'])[0]
    end_logits = interpreter.get_tensor(output_details[1]['index'])[0]
    return start_logits, end_logits, inference_time_ms

# Extract answer text from logits
def get_answer(start_logits, end_logits, input_ids):
    start = np.argmax(start_logits)
    end = np.argmax(end_logits)
    if end < start or (end - start + 1) > 30:
        return ""
    tokens = input_ids[0][start:end+1]
    return tokenizer.decode(tokens, skip_special_tokens=True)

# Run inference loop and log results
for i in range(len(test_samples)):
    context, question, true_answer, _ = test_samples[i]
    input_ids, attention_mask, segment_ids = encode(question, context)

    mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    start_logits, end_logits, inf_time = predict(input_ids, attention_mask, segment_ids)
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB

    mem_used = mem_after - mem_before

    predicted_answer = get_answer(start_logits, end_logits, input_ids)
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")

    # Placeholder zeros for metrics not yet tracked
    row = [
        timestamp,
        predicted_answer,      # Review
        "inference",          # Mode
        0,                    # Pre_Lat_ms
        round(inf_time, 2),   # Inf_Lat_ms (inference time)
        0,                    # Post_Lat_ms
        0, 0, 0,              # Pre_E_mJ, Inf_E_mJ, Post_E_mJ
        0, 0, 0, 0,           # Pre_Max_V, Pre_Mean_V, Pre_Max_C, Pre_Mean_C
        0, 0, 0, 0,           # Inf_Max_V, Inf_Mean_V, Inf_Max_C, Inf_Mean_C
        0, 0, 0, 0,           # Post_Max_V, Post_Mean_V, Post_Max_C, Post_Mean_C
        0, 0, 0                # Pre_Pwr_mW, Inf_Pwr_mW, Post_Pwr_mW
    ]

    append_csv_row(row)

    print(f"Logged sample {i+1} at {timestamp}")
    print(f"Q: {question}")
    print(f"Predicted: {predicted_answer}")
    print(f"True: {true_answer}")
    print(f"Memory used: {mem_used:.4f} MB, Inference time: {inf_time:.2f} ms\n")

print(f"Log saved to {OUTPUT_PATH}")

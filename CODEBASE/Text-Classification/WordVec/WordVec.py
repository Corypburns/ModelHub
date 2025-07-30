import tensorflow as tf
import time as t
import time  # For perf_counter and sleep
from datetime import datetime as dt
from pathlib import Path
import keyboard as k
import platform, numpy as np
import pandas as pd

# === CONFIG ===
if platform.system() == "Windows":
    BASE_PATH = Path(r"E:\Code\Python\ModelHub")
else:
    BASE_PATH = Path.home() / "ModelHub"

MODEL_PATH = BASE_PATH / "MODELBASE" / "Text-Classification" / "wordvec.tflite"
DATASET_PATH = BASE_PATH / "DATASETS" / "Text-Classification" / "WordVec" / "IMDB_Dataset.csv"
LOG_DIR = BASE_PATH / "OUTPUTS" / "Text-Classification" / "WordVec"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_WV_Text-Classification_{DATE_TIME}.csv"
OUTPUT_PATH = LOG_DIR / FILE_NAME
HEADERS = (
    "Timestamp,Review,Mode,"
    "Pre_Lat_ms,Inf_Lat_ms,Post_Lat_ms,"
    "Pre_E_mJ,Inf_E_mJ,Post_E_mJ,"
    "Pre_Max_V,Pre_Mean_V,Pre_Max_C,Pre_Mean_C,"
    "Inf_Max_V,Inf_Mean_V,Inf_Max_C,Inf_Mean_C,"
    "Post_Max_V,Post_Mean_V,Post_Max_C,Post_Mean_C,"
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW\n"
)

# === UTILITY METHODS ===            
def append_csv_row(*args):
    row = ",".join(map(str, args)) + "\n"
    with open(OUTPUT_PATH, 'a') as f:
        f.write(row)
        
def read_dataset():
    data = pd.read_csv(DATASET_PATH)
    if not OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'w') as f:
            f.write(HEADERS)
            
    with open(OUTPUT_PATH, 'a') as f:
        append_csv_row(DATE_TIME, data.head(1))

    return data

# === Model Load & Allocation ===
def load_and_allocate():
    start_load = t.time()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    end_load = t.time()
    
    start_allocation = t.time()
    interpreter.allocate_tensors()
    end_allocation = t.time()
    
    print(f"\nModel loaded in {(end_load - start_load) * 1000:.2f} ms.", 
          f"\nTensors allocated in {(end_allocation - start_allocation) * 1000:.2f} ms.",
          f"\nPress 'Space' to continue...")
    k.wait('space')
    
    return interpreter

def text_tokenizer():
    data = read_dataset()
    unique_vocab = []
    for review in data['review']:
        split_vocab = review.split()  # Words 
        unique_vocab.extend(split_vocab)  # Gets all the words
    uniques = sorted(set(unique_vocab))  # Gets all unique words
        
    vocab = {word: i for i, word in enumerate(uniques)}
    
    return vocab

def truncation(seq, max_length: int, pad_value: int = 0, direction: str = "post"):
    if len(seq) > max_length:
        return seq[:max_length] if direction == "post" else seq[-max_length:]
    else:
        return seq + [pad_value] * (max_length - len(seq)) if direction == "post" else [pad_value] * (max_length - len(seq)) + seq

def text_classification_step(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    max_vocab_length = 9999  # Adjust based on your model's vocab size
    data = read_dataset()
    vocab = text_tokenizer()
    max_length = input_details[0]['shape'][1]

    for review in data['review']:
        tokens = review.lower().split()
        id_seq = [min(vocab.get(word, 0), max_vocab_length) for word in tokens]
        id_seq = truncation(id_seq, max_length, direction="pre")
        input_array = np.array([id_seq], dtype=np.int32)

        interpreter.set_tensor(input_details[0]['index'], input_array)

        start_inf = time.perf_counter()
        interpreter.invoke()
        end_inf = time.perf_counter()

        outputs = interpreter.get_tensor(output_details[0]['index'])

        inf_time_ms = (end_inf - start_inf) * 1000

        predicted_class = np.argmax(outputs)
        confidence = outputs[0][predicted_class] if outputs.ndim == 2 else outputs[predicted_class]

        print(f"Review: {review}\n")
        print(f"Prediction: Class {predicted_class} with confidence {(confidence * 100):.2f}%")
        print(f"Inference time: {inf_time_ms:.5f} ms\n")

        time.sleep(2)  # Pause 2 seconds so you can read output comfortably

def main():
    interpreter = load_and_allocate()
    text_classification_step(interpreter)

main()

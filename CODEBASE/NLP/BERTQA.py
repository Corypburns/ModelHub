import tensorflow as tf
import numpy as np
from transformers import BertTokenizer as BT
import json
from sklearn.model_selection import train_test_split as tts
import os, time as t
from datetime import datetime as dt
import socket
from pathlib import Path
from jtop import jtop

# === HOST SETUP ===
host = socket.gethostname()
match host:
    case "DESKTOP-FI8GT7F":
        BASE_PATH = Path(r"E:\Code\Python\ModelHub")
    case "nano1-desktop":
        BASE_PATH = Path("/home/nano1/anik-lab/ModelHub")
    case _:
        raise RuntimeError("Unknown host - configure BASE_PATH.")

# === PATHS ===
TRAIN_PATH = BASE_PATH / 'DATASETS' / 'NLP' / 'Train' / 'train-v2.0.json'
MODEL_PATH = BASE_PATH / 'MODELBASE' / 'NLP' / 'MobileBert.tflite'
LOG_DIR = BASE_PATH / 'OUTPUTS' / 'NLP'
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_bertqa_jetson_{DATE_TIME}.csv"
OUTPUT_PATH = LOG_DIR / FILE_NAME

# === CSV SETUP ===
HEADERS = (
    "Timestamp,Review,Mode,"
    "Pre_Lat_ms,Inf_Lat_ms,Post_Lat_ms,"
    "Pre_E_mJ,Inf_E_mJ,Post_E_mJ,"
    "Pre_Max_V,Pre_Mean_V,Pre_Max_C,Pre_Mean_C,"
    "Inf_Max_V,Inf_Mean_V,Inf_Max_C,Inf_Mean_C,"
    "Post_Max_V,Post_Mean_V,Post_Max_C,Post_Mean_C,"
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW\n"
)

def init_csv():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
            f.write(HEADERS)

def append_csv_row(*args):
    row = ",".join(map(str, args)) + "\n"
    with open(OUTPUT_PATH, 'a') as f:
        f.write(row)

# === JETSON METRICS ===
def get_total_power(jt):
    p = jt.power
    if "tot" in p: return float(p["tot"]["power"])
    if "total" in p: return float(p["total"]["power"])
    if "rail" in p:
        rails = [r for r in p["rail"].values() if isinstance(r, dict)]
        return sum(float(r.get("power", 0)) for r in rails) / len(rails) if rails else 0.0
    return 0.0

def get_voltage_current_stats(jt):
    rails = jt.power.get("rail", {}).values()
    vs, cs = [], []
    for r in rails:
        try:
            vs.append(float(r.get("volt", 0)))
            cs.append(float(r.get("curr", 0)))
        except:
            continue
    return (max(vs) if vs else 0, sum(vs)/len(vs) if vs else 0,
            max(cs) if cs else 0, sum(cs)/len(cs) if cs else 0)

def get_jetson_stats(jt):
    pwr = get_total_power(jt)
    max_v, mean_v, max_c, mean_c = get_voltage_current_stats(jt)
    return max_v, mean_v, max_c, mean_c, pwr

# === MODEL PREP ===
tokenizer = BT.from_pretrained("google/mobilebert-uncased")

with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
sample_list = [(qa['question'], paragraph['context'], qa['answers'][0]['text'] if qa['answers'] else "")
               for article in data['data'] for paragraph in article['paragraphs'] for qa in paragraph['qas']]
_, test_samples = tts(sample_list, test_size=0.1, random_state=42)

def encode(question, context, max_len=384):
    tokens = tokenizer.encode_plus(
        question, context,
        max_length=max_len, padding='max_length', truncation=True,
        return_tensors='np'
    )
    return (tokens['input_ids'].astype(np.int32),
            tokens['attention_mask'].astype(np.int32),
            tokens['token_type_ids'].astype(np.int32))

def predict(interpreter, inputs):
    input_ids, attn_mask, token_type_ids = inputs
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_ids)
    interpreter.set_tensor(input_details[1]['index'], attn_mask)
    interpreter.set_tensor(input_details[2]['index'], token_type_ids)
    interpreter.invoke()
    start_logits = interpreter.get_tensor(output_details[0]['index'])[0]
    end_logits = interpreter.get_tensor(output_details[1]['index'])[0]
    return start_logits, end_logits

def get_answer(start_logits, end_logits, input_ids):
    start = np.argmax(start_logits)
    end = np.argmax(end_logits)
    if end < start or (end - start + 1) > 30:
        return ""
    tokens = input_ids[0][start:end+1]
    return tokenizer.decode(tokens, skip_special_tokens=True)

def run_pipeline(num_threads, mode_label):
    print(f"\nLoading model with {mode_label} (threads={num_threads})...")
    init_csv()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=num_threads)
    interpreter.allocate_tensors()
    print("Model ready.\n")

    with jtop() as jetson:
        for i, (question, context, true_answer) in enumerate(test_samples[:1000]):
            timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
            # PRE
            pre_max_v, pre_mean_v, pre_max_c, pre_mean_c, pre_pwr = get_jetson_stats(jetson)
            t0 = t.time(); inputs = encode(question, context); t1 = t.time()
            pre_lat = (t1 - t0)*1000; pre_e = pre_pwr*(t1 - t0)
            # INF
            inf_max_v_start, inf_mean_v_start, inf_max_c_start, inf_mean_c_start, inf_pwr_start = get_jetson_stats(jetson)
            t2 = t.time(); sl, el = predict(interpreter, inputs); t3 = t.time()
            inf_max_v_end, inf_mean_v_end, inf_max_c_end, inf_mean_c_end, inf_pwr_end = get_jetson_stats(jetson)
            inf_lat = (t3 - t2)*1000
            inf_e = ((inf_pwr_start+inf_pwr_end)/2)*(t3 - t2)
            inf_max_v = max(inf_max_v_start, inf_max_v_end)
            inf_mean_v = (inf_mean_v_start+inf_mean_v_end)/2
            inf_max_c = max(inf_max_c_start, inf_max_c_end)
            inf_mean_c = (inf_mean_c_start+inf_mean_c_end)/2
            inf_pwr = (inf_pwr_start+inf_pwr_end)/2
            # POST
            t4 = t.time(); pred = get_answer(sl, el, inputs[0]); t5 = t.time()
            post_max_v, post_mean_v, post_max_c, post_mean_c, post_pwr = get_jetson_stats(jetson)
            post_lat = (t5 - t4)*1000; post_e = post_pwr*(t5 - t4)
            # Print
            print(f"[{i+1:3d}] Q: {question}\nPredicted: {pred}\nActual   : {true_answer}\n"
                  f"Lat (ms): Pre={pre_lat:.2f} Inf={inf_lat:.2f} Post={post_lat:.2f}\n"
                  f"Energy (mJ): Pre={pre_e:.2f} Inf={inf_e:.2f} Post={post_e:.2f}\n"
                  f"Pwr (mW): Pre={pre_pwr:.2f} Inf={inf_pwr:.2f} Post={post_pwr:.2f}\n"
                  f"Volt (V): Pre={pre_max_v:.2f} Inf={inf_mean_v:.2f} Post={post_max_v:.2f}\n"
                  f"Curr (A): Pre={pre_max_c:.2f} Inf={inf_mean_c:.2f} Post={post_max_c:.2f}\n")
            # Log
            append_csv_row(timestamp, question, mode_label, pre_lat, inf_lat, post_lat,
                           pre_e, inf_e, post_e, pre_max_v, pre_mean_v, pre_max_c, pre_mean_c,
                           inf_max_v, inf_mean_v, inf_max_c, inf_mean_c,
                           post_max_v, post_mean_v, post_max_c, post_mean_c,
                           pre_pwr, inf_pwr, post_pwr)
            t.sleep(1)

def menu():
    print("\n--- Select Mode ---")
    print("1: CPU - 1 Thread")
    print("2: CPU - 4 Threads")
    print("3: GPU (if supported)")
    print("4: Quit")
    choice = input("Enter your choice: ").strip()
    if choice == '1':
        run_pipeline(num_threads=1, mode_label="CPU1")
    elif choice == '2':
        run_pipeline(num_threads=4, mode_label="CPU4")
    elif choice == '3':
        # GPU requires special delegate; example placeholder
        print("GPU mode selected (implement delegate as needed).")
        run_pipeline(num_threads=1, mode_label="GPU")
    elif choice == '4':
        print("Exiting...")
        exit()
    else:
        print("Invalid choice, try again.")
        menu()

if __name__ == "__main__":
    menu()


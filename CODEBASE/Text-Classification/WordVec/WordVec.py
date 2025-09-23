import tensorflow as tf
import time as t
from datetime import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import socket
from jtop import jtop   # Jetson stats

# === CONFIG ===
host = socket.gethostname()
print("Running on host:", host)

match host:
    case "DESKTOP-FI8GT7F":
        BASE_PATH = Path(r"E:\Code\Python\ModelHub")
    case "CoryPC":
        BASE_PATH = None
    case "nano1-desktop":
        BASE_PATH = Path("/home/nano1/anik-lab/ModelHub")

MODEL_PATH = BASE_PATH / "MODELBASE" / "Text-Classification" / "wordvec.tflite"
DATASET_PATH = BASE_PATH / "DATASETS" / "Text-Classification" / "WordVec" / "IMDB_Dataset.csv"
LOG_DIR = BASE_PATH / "OUTPUTS" / "Text-Classification" / "WordVec"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_WV_Text-Classification_{DATE_TIME}.csv"
OUTPUT_PATH = LOG_DIR / FILE_NAME

# === CSV HEADER (EfficientNet style) ===
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
        with open(OUTPUT_PATH, "w") as f:
            f.write(HEADERS)

def append_csv_row(
    timestamp, review, mode,
    pre_lat_ms, inf_lat_ms, post_lat_ms,
    pre_e_mJ, inf_e_mJ, post_e_mJ,
    pre_max_v, pre_mean_v, pre_max_c, pre_mean_c,
    inf_max_v, inf_mean_v, inf_max_c, inf_mean_c,
    post_max_v, post_mean_v, post_max_c, post_mean_c,
    pre_pwr, inf_pwr, post_pwr, memory
):
    row = ",".join(map(str, [
        timestamp, review, mode,
        f"{pre_lat_ms:.4f}", f"{inf_lat_ms:.4f}", f"{post_lat_ms:.4f}",
        f"{pre_e_mJ:.4f}", f"{inf_e_mJ:.4f}", f"{post_e_mJ:.4f}",
        f"{pre_max_v:.4f}", f"{pre_mean_v:.4f}", f"{pre_max_c:.4f}", f"{pre_mean_c:.4f}",
        f"{inf_max_v:.4f}", f"{inf_mean_v:.4f}", f"{inf_max_c:.4f}", f"{inf_mean_c:.4f}",
        f"{post_max_v:.4f}", f"{post_mean_v:.4f}", f"{post_max_c:.4f}", f"{post_mean_c:.4f}",
        f"{pre_pwr:.4f}", f"{inf_pwr:.4f}", f"{post_pwr:.4f}"
    ])) + "\n"
    with open(OUTPUT_PATH, "a") as f:
        f.write(row)

# === Jetson Stats ===
def get_total_power(jt):
    p = jt.power
    for k in ("tot", "total", "Total"):
        if k in p:
            v = p[k]
            return float(v["power"] if isinstance(v, dict) else v)
    if "rail" in p:
        vals = [float(r.get("power", 0)) for r in p["rail"].values() if isinstance(r, dict)]
        return sum(vals) / len(vals) if vals else 0.0
    return 0.0

def get_voltage_current_stats(jt):
    rails = jt.power.get("rail", {}).values()
    vs, cs = [], []
    for r in rails:
        try:
            vs.append(float(r.get("volt", 0)))
            cs.append(float(r.get("curr", 0)))
        except:
            pass
    return (max(vs) if vs else 0, sum(vs)/len(vs) if vs else 0,
            max(cs) if cs else 0, sum(cs)/len(cs) if cs else 0)

def get_jetson_stats(jt):
    pwr = get_total_power(jt)
    max_v, mean_v, max_c, mean_c = get_voltage_current_stats(jt)
    return max_v, mean_v, max_c, mean_c, pwr

# === DATA + MODEL ===
def read_dataset():
    return pd.read_csv(DATASET_PATH)

def load_model(num_threads):
    start_load = t.time()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=num_threads)
    end_load = t.time()
    start_alloc = t.time()
    interpreter.allocate_tensors()
    end_alloc = t.time()

    print(f"Model load time: {(end_load - start_load)*1000:.2f} ms,"
          f" allocation: {(end_alloc - start_alloc)*1000:.2f} ms.")
    input("Press Enter to continue...")
    return interpreter

def text_tokenizer(data):
    vocab = {word: i for i, word in enumerate(sorted(set(" ".join(data['review']).split())))}
    return vocab

def truncation(seq, max_length: int, pad_value: int = 0, direction: str = "post"):
    if len(seq) > max_length:
        return seq[:max_length] if direction == "post" else seq[-max_length:]
    else:
        return seq + [pad_value]*(max_length - len(seq)) if direction=="post" else [pad_value]*(max_length-len(seq)) + seq

# === INFERENCE ===
def text_classification_step(interpreter, mode="CPU1"):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    max_vocab_length = 9999
    data = read_dataset()
    vocab = text_tokenizer(data)
    max_length = input_details[0]['shape'][1]

    with jtop() as jetson:
        for idx, review in enumerate(data['review'][:1000], start=1):
            # Pre phase
            pre_max_v, pre_mean_v, pre_max_c, pre_mean_c, pre_pwr = get_jetson_stats(jetson)
            pre_time = t.time()

            tokens = review.lower().split()
            id_seq = [min(vocab.get(w,0), max_vocab_length) for w in tokens]
            id_seq = truncation(id_seq, max_length, direction="pre")
            input_array = np.array([id_seq], dtype=np.int32)

            # Inference
            inf_max_v_start, inf_mean_v_start, inf_max_c_start, inf_mean_c_start, inf_pwr_start = get_jetson_stats(jetson)
            inf_start = t.time()
            interpreter.set_tensor(input_details[0]['index'], input_array)
            interpreter.invoke()
            inf_end = t.time()
            inf_max_v_end, inf_mean_v_end, inf_max_c_end, inf_mean_c_end, inf_pwr_end = get_jetson_stats(jetson)

            # Post
            post_time = t.time()
            post_max_v, post_mean_v, post_max_c, post_mean_c, post_pwr = get_jetson_stats(jetson)

            stats = jstson.stats
            ram_stats = stats['RAM']

            if isinstance(ram_stats, dict):
                ram_used = float(ram_stats.get('used', 0))
            else:
                ram_used = float(ram_stats)

            # Results
            outputs = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(outputs)
            confidence = outputs[0][predicted_class] if outputs.ndim==2 else outputs[predicted_class]

            # Latency
            pre_lat = (inf_start - pre_time) * 1000
            inf_lat = (inf_end - inf_start) * 1000
            post_lat = (post_time - inf_end) * 1000

            # Energy
            pre_energy = pre_pwr * (inf_start - pre_time)
            inf_energy = ((inf_pwr_start+inf_pwr_end)/2) * (inf_end - inf_start)
            post_energy = post_pwr * (post_time - inf_end)

            # Inference averaged stats
            inf_max_v = max(inf_max_v_start, inf_max_v_end)
            inf_mean_v = (inf_mean_v_start+inf_mean_v_end)/2
            inf_max_c = max(inf_max_c_start, inf_max_c_end)
            inf_mean_c = (inf_mean_c_start+inf_mean_c_end)/2
            inf_pwr = (inf_pwr_start+inf_pwr_end)/2

            # === Print summary (EfficientNet-style) ===
            print(f"\n--- Inference Stats ---")
            print(f"Review #{idx}")
            print(f"{review}")
            print(f"Prediction: Class {predicted_class} ({confidence*100:.2f}%)\n")
            print(f" Latencies (ms):\n"
                  f"    Pre-processing : {pre_lat:.2f}\n"
                  f"    Inference      : {inf_lat:.2f}\n"
                  f"    Post-processing: {post_lat:.2f}\n")
            print(f" Energy (mJ):\n"
                  f"    Pre-processing : {pre_energy:.2f}\n"
                  f"    Inference      : {inf_energy:.2f}\n"
                  f"    Post-processing: {post_energy:.2f}\n")
            print(f" Power (mW):\n"
                  f"    Pre-processing : {pre_pwr:.2f}\n"
                  f"    Inference      : {inf_pwr:.2f}\n"
                  f"    Post-processing: {post_pwr:.2f}\n")
            print(f" Voltage (V):\n"
                  f"    Pre-processing : {pre_max_v:.2f}\n"
                  f"    Inference      : {inf_mean_v:.2f}\n"
                  f"    Post-processing: {post_max_v:.2f}\n")
            print(f" Current (A):\n"
                  f"    Pre-processing : {pre_max_c:.2f}\n"
                  f"    Inference      : {inf_mean_c:.2f}\n"
                  f"    Post-processing: {post_max_c:.2f}\n")
            print(f" Memory Consumption:\n"
                  f"    Memory : {ram_stats}")

            # === Log to CSV (review as index) ===
            append_csv_row(
                timestamp=dt.now().strftime("%Y-%m-%dT%H:%M:%S"),
                review=str(idx),
                mode=mode,
                pre_lat_ms=pre_lat, inf_lat_ms=inf_lat, post_lat_ms=post_lat,
                pre_e_mJ=pre_energy, inf_e_mJ=inf_energy, post_e_mJ=post_energy,
                pre_max_v=pre_max_v, pre_mean_v=pre_mean_v, pre_max_c=pre_max_c, pre_mean_c=pre_mean_c,
                inf_max_v=inf_max_v, inf_mean_v=inf_mean_v, inf_max_c=inf_max_c, inf_mean_c=inf_mean_c,
                post_max_v=post_max_v, post_mean_v=post_mean_v, post_max_c=post_max_c, post_mean_c=post_mean_c,
                pre_pwr=pre_pwr, inf_pwr=inf_pwr, post_pwr=post_pwr, memory=ram_stats
            )

            t.sleep(1)

# === MENU ===
def menu():
    print("\n--- Select Mode ---\n1: CPU 1 Thread\n2: CPU 4 Threads\n3: GPU\n4: Quit\n")
    choice = input("Enter your choice: ").strip()
    if choice == '1':
        interpreter = load_model(num_threads=1)
        text_classification_step(interpreter, mode="CPU1")
    elif choice == '2':
        interpreter = load_model(num_threads=4)
        text_classification_step(interpreter, mode="CPU4")
    elif choice == '3':
        interpreter = load_model(num_threads=0)
        text_classification_step(interpreter, mode="GPU")
    elif choice == '4':
        print("Exiting...")
        exit()
    else:
        print("Invalid choice, try again.")
        menu()

# === MAIN ===
def main():
    init_csv()
    menu()

if __name__ == "__main__":
    main()


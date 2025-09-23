import tensorflow as tf
import time as t
from datetime import datetime as dt
from pathlib import Path
import numpy as np
import json
import socket
from transformers import BertTokenizer
from jtop import jtop

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

MODEL_PATH = BASE_PATH / "MODELBASE" / "NLP" / "MobileBert.tflite"
DATASET_PATH = BASE_PATH / "DATASETS" / "NLP" / "Train" / "train-v2.0.json"
LOG_DIR = BASE_PATH / "OUTPUTS" / "NLP"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_BERTQA_{DATE_TIME}.csv"
OUTPUT_PATH = LOG_DIR / FILE_NAME

# === CSV HEADER ===
HEADERS = (
    "Timestamp,Review,Mode,"
    "Pre_Lat_ms,Inf_Lat_ms,Post_Lat_ms,"
    "Pre_E_mJ,Inf_E_mJ,Post_E_mJ,"
    "Pre_Max_V,Pre_Mean_V,Pre_Max_C,Pre_Mean_C,"
    "Inf_Max_V,Inf_Mean_V,Inf_Max_C,Inf_Mean_C,"
    "Post_Max_V,Post_Mean_V,Post_Max_C,Post_Mean_C,"
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW,Memory_mB\n"
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
        f"{pre_pwr:.4f}", f"{inf_pwr:.4f}", f"{post_pwr:.4f}", f"{memory:.2f}"
    ])) + "\n"
    with open(OUTPUT_PATH, "a") as f:
        f.write(row)

# === Jetson Stats ===
def get_total_power(jt):
    p = jt.power
    for k in ("tot","total","Total"):
        if k in p:
            v = p[k]
            return float(v["power"] if isinstance(v, dict) else v)
    if "rail" in p:
        vals = [float(r.get("power",0)) for r in p["rail"].values() if isinstance(r, dict)]
        return sum(vals)/len(vals) if vals else 0.0
    return 0.0

def get_voltage_current_stats(jt):
    rails = jt.power.get("rail", {}).values()
    vs, cs = [], []
    for r in rails:
        try:
            vs.append(float(r.get("volt",0)))
            cs.append(float(r.get("curr",0)))
        except:
            pass
    return (max(vs) if vs else 0, sum(vs)/len(vs) if vs else 0,
            max(cs) if cs else 0, sum(cs)/len(cs) if cs else 0)

def get_jetson_stats(jt):
    pwr = get_total_power(jt)
    max_v, mean_v, max_c, mean_c = get_voltage_current_stats(jt)
    return max_v, mean_v, max_c, mean_c, pwr

# === LOAD DATA & MODEL ===
def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [(qa['question'], paragraph['context'], qa['answers'][0]['text'] if qa['answers'] else "")
               for article in data['data']
               for paragraph in article['paragraphs']
               for qa in paragraph['qas']]
    return samples[:1000]  # limit for demo

def load_model(num_threads):
    start_load = t.time()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=num_threads)
    end_load = t.time()
    start_alloc = t.time()
    interpreter.allocate_tensors()
    end_alloc = t.time()
    print(f"Model load time: {(end_load-start_load)*1000:.2f} ms, allocation: {(end_alloc-start_alloc)*1000:.2f} ms")
    input("Press Enter to continue...")
    return interpreter

tokenizer = BertTokenizer.from_pretrained("google/mobilebert-uncased")

def encode(question, context, max_len=384):
    tokens = tokenizer.encode_plus(question, context, max_length=max_len, padding="max_length",
                                   truncation=True, return_tensors="np")
    return tokens['input_ids'].astype(np.int32), tokens['attention_mask'].astype(np.int32), tokens['token_type_ids'].astype(np.int32)

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

# === INFERENCE ===
def bertqa_step(interpreter, samples, mode="CPU1"):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with jtop() as jetson:
        for idx, (question, context, true_answer) in enumerate(samples, start=1):
            # Pre
            pre_max_v, pre_mean_v, pre_max_c, pre_mean_c, pre_pwr = get_jetson_stats(jetson)
            pre_time = t.time()
            inputs = encode(question, context)
            pre_lat = (t.time()-pre_time)*1000
            pre_energy = pre_pwr*(pre_lat/1000)

            # Inference
            inf_max_v_start, inf_mean_v_start, inf_max_c_start, inf_mean_c_start, inf_pwr_start = get_jetson_stats(jetson)
            inf_start = t.time()
            start_logits, end_logits = predict(interpreter, inputs)
            inf_end = t.time()
            inf_max_v_end, inf_mean_v_end, inf_max_c_end, inf_mean_c_end, inf_pwr_end = get_jetson_stats(jetson)
            inf_lat = (inf_end-inf_start)*1000
            inf_energy = ((inf_pwr_start+inf_pwr_end)/2)*(inf_lat/1000)
            inf_max_v = max(inf_max_v_start, inf_max_v_end)
            inf_mean_v = (inf_mean_v_start+inf_mean_v_end)/2
            inf_max_c = max(inf_max_c_start, inf_max_c_end)
            inf_mean_c = (inf_mean_c_start+inf_mean_c_end)/2
            inf_pwr = (inf_pwr_start+inf_pwr_end)/2

            # Post
            post_time = t.time()
            pred_answer = get_answer(start_logits, end_logits, inputs[0])
            post_max_v, post_mean_v, post_max_c, post_mean_c, post_pwr = get_jetson_stats(jetson)
            post_lat = (t.time()-post_time)*1000
            post_energy = post_pwr*(post_lat/1000)

            # Memory
            ram_stats = jetson.stats['RAM']
            memory = float(ram_stats['used']) if isinstance(ram_stats, dict) else float(ram_stats)

            # Print
            print(f"\n--- Inference Stats ---\nSample #{idx}\nQ: {question}\nPrediction: {pred_answer}\nTrue: {true_answer}\n")
            print(f"Latencies (ms): Pre={pre_lat:.2f} Inf={inf_lat:.2f} Post={post_lat:.2f}")
            print(f"Energy (mJ): Pre={pre_energy:.2f} Inf={inf_energy:.2f} Post={post_energy:.2f}")
            print(f"Power (mW): Pre={pre_pwr:.2f} Inf={inf_pwr:.2f} Post={post_pwr:.2f}")
            print(f"Voltage (V): Pre={pre_max_v:.2f} Inf={inf_mean_v:.2f} Post={post_max_v:.2f}")
            print(f"Current (A): Pre={pre_max_c:.2f} Inf={inf_mean_c:.2f} Post={post_max_c:.2f}")
            print(f"Memory (MB): {memory:.2f}\n")

            # Log
            append_csv_row(
                timestamp=dt.now().strftime("%Y-%m-%dT%H:%M:%S"),
                review=str(idx),
                mode=mode,
                pre_lat_ms=pre_lat, inf_lat_ms=inf_lat, post_lat_ms=post_lat,
                pre_e_mJ=pre_energy, inf_e_mJ=inf_energy, post_e_mJ=post_energy,
                pre_max_v=pre_max_v, pre_mean_v=pre_mean_v, pre_max_c=pre_max_c, pre_mean_c=pre_mean_c,
                inf_max_v=inf_max_v, inf_mean_v=inf_mean_v, inf_max_c=inf_max_c, inf_mean_c=inf_mean_c,
                post_max_v=post_max_v, post_mean_v=post_mean_v, post_max_c=post_max_c, post_mean_c=post_mean_c,
                pre_pwr=pre_pwr, inf_pwr=inf_pwr, post_pwr=post_pwr, memory=memory
            )

            t.sleep(1)

# === MENU ===
def menu():
    samples = load_dataset()
    print("\n--- Select Mode ---\n1: CPU 1 Thread\n2: CPU 4 Threads\n3: GPU\n4: Quit\n")
    choice = input("Enter your choice: ").strip()
    if choice=='1':
        interpreter = load_model(num_threads=1)
        bertqa_step(interpreter, samples, mode="CPU1")
    elif choice=='2':
        interpreter = load_model(num_threads=4)
        bertqa_step(interpreter, samples, mode="CPU4")
    elif choice=='3':
        interpreter = load_model(num_threads=0)
        bertqa_step(interpreter, samples, mode="GPU")
    elif choice=='4':
        print("Exiting...")
        exit()
    else:
        print("Invalid choice, try again.")
        menu()

# === MAIN ===
def main():
    init_csv()
    menu()

if __name__=="__main__":
    main()

from datetime import datetime as dt
from pathlib import Path
import tensorflow as tf
import time as t
import socket
import cv2
import numpy as np
from jtop import jtop

# === HOST CONFIG ===
host = socket.gethostname()

match host:
    case "DESKTOP-FI8GT7F":
        BASE_PATH = Path(r"E:\Code\Python\ModelHub")
        TEST_IMAGE_BASE_PATH = Path(r"E:\Code\Python\DATASETS\COCO")
    case "CoryPC":
        BASE_PATH = None
        TEST_IMAGE_BASE_PATH = None
    case "nano1-desktop":
        BASE_PATH = Path("/home/nano1/anik-lab/ModelHub")
        TEST_IMAGE_BASE_PATH = Path("/home/nano1/anik-lab/coco")

TEST_IMAGE_PATH = TEST_IMAGE_BASE_PATH / "test2017"
MODEL_PATH = BASE_PATH / "MODELBASE" / "Image-Classification" / "EfficientNet_lite4-224x224_Quantized" / "efficientnet_lite4_int8_2.tflite"
LOG_DIR = BASE_PATH / "OUTPUTS" / "Image-Classification" / "EfficientNet_lite4-224x224_Quantized"
LABEL_MAP = BASE_PATH / "LABELMAPS" / "Image-Classification" / "labels.txt"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_ENL4_224x224_quantized_{DATE_TIME}.csv"
OUTPUT_PATH = LOG_DIR / FILE_NAME

HEADERS = (
    "Timestamp,Review,Mode,"
    "Pre_Lat_ms,Inf_Lat_ms,Post_Lat_ms,"
    "Pre_E_mJ,Inf_E_mJ,Post_E_mJ,"
    "Pre_Max_V,Pre_Mean_V,Pre_Max_C,Pre_Mean_C,"
    "Inf_Max_V,Inf_Mean_V,Inf_Max_C,Inf_Mean_C,"
    "Post_Max_V,Post_Mean_V,Post_Max_C,Post_Mean_C,"
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW,Memory_mB\n"
)

# === CSV UTILITY ===
def init_csv():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'w') as f:
            f.write(HEADERS)

def append_csv_row(timestamp, review, mode,
                   pre_lat_ms, inf_lat_ms, post_lat_ms,
                   pre_e_mJ, inf_e_mJ, post_e_mJ,
                   pre_max_v, pre_mean_v, pre_max_c, pre_mean_c,
                   inf_max_v, inf_mean_v, inf_max_c, inf_mean_c,
                   post_max_v, post_mean_v, post_max_c, post_mean_c,
                   pre_pwr, inf_pwr, post_pwr, memory):
    row = ",".join(map(str, [
        timestamp, review, mode,
        f"{pre_lat_ms:.4f}", f"{inf_lat_ms:.4f}", f"{post_lat_ms:.4f}",
        f"{pre_e_mJ:.4f}", f"{inf_e_mJ:.4f}", f"{post_e_mJ:.4f}",
        f"{pre_max_v:.4f}", f"{pre_mean_v:.4f}", f"{pre_max_c:.4f}", f"{pre_mean_c:.4f}",
        f"{inf_max_v:.4f}", f"{inf_mean_v:.4f}", f"{inf_max_c:.4f}", f"{inf_mean_c:.4f}",
        f"{post_max_v:.4f}", f"{post_mean_v:.4f}", f"{post_max_c:.4f}", f"{post_mean_c:.4f}",
        f"{pre_pwr:.4f}", f"{inf_pwr:.4f}", f"{post_pwr:.4f}", f"{memory:.4f}"
    ])) + "\n"
    with open(OUTPUT_PATH, 'a') as f:
        f.write(row)

# === LABELS & MODEL ===
def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def load_model(num_threads):
    start_load = t.time()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=num_threads)
    end_load = t.time()

    start_alloc = t.time()
    interpreter.allocate_tensors()
    end_alloc = t.time()

    print(f"Model load time: {(end_load - start_load)*1000:.2f} ms")
    print(f"Model allocation time: {(end_alloc - start_alloc)*1000:.2f} ms")
    input("Press Enter to continue...")
    return interpreter

# === JETSON STATS ===
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
    rails = jt.power.get("rail",{}).values()
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

# === IMAGE PROCESSING & INFERENCE ===
def run_inference(interpreter, labels, mode="CPU1"):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]

    with jtop() as jetson:
        for img_path in TEST_IMAGE_PATH.glob("*.jpg"):
            pre_max_v, pre_mean_v, pre_max_c, pre_mean_c, pre_pwr = get_jetson_stats(jetson)
            pre_time = t.time()

            raw_img = cv2.imread(str(img_path))
            resized_img = cv2.resize(raw_img, (width,height))
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(rgb_img.astype(np.uint8), axis=0)

            inf_max_v_start, inf_mean_v_start, inf_max_c_start, inf_mean_c_start, inf_pwr_start = get_jetson_stats(jetson)
            inf_start = t.time()

            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()

            inf_end = t.time()
            inf_max_v_end, inf_mean_v_end, inf_max_c_end, inf_mean_c_end, inf_pwr_end = get_jetson_stats(jetson)
            post_time = t.time()
            post_max_v, post_mean_v, post_max_c, post_mean_c, post_pwr = get_jetson_stats(jetson)

            # --- Memory Logging ---
            ram_stats = jetson.stats.get('RAM', {})
            memory = float(ram_stats.get('used', 0)) if isinstance(ram_stats, dict) else 0.0

            output = interpreter.get_tensor(output_details[0]['index'])[0]
            predicted_index = output.argmax()
            predicted_label = labels[predicted_index]
            confidence = output[predicted_index]

            pre_lat = (inf_start-pre_time)*1000
            inf_lat = (inf_end-inf_start)*1000
            post_lat = (post_time-inf_end)*1000

            pre_energy = pre_pwr*(inf_start-pre_time)
            inf_energy = ((inf_pwr_start+inf_pwr_end)/2)*(inf_end-inf_start)
            post_energy = post_pwr*(post_time-inf_end)

            inf_max_v = max(inf_max_v_start, inf_max_v_end)
            inf_mean_v = (inf_mean_v_start+inf_mean_v_end)/2
            inf_max_c = max(inf_max_c_start, inf_max_c_end)
            inf_mean_c = (inf_mean_c_start+inf_mean_c_end)/2
            inf_pwr = (inf_pwr_start+inf_pwr_end)/2

            print(f"\nImage: {img_path.name} | Pred: {predicted_label} ({confidence*100:.2f}%) | Inf Lat: {inf_lat:.2f} ms | Memory: {memory:.2f} MB")

            append_csv_row(
                timestamp=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                review=img_path.name,
                mode=mode,
                pre_lat_ms=pre_lat, inf_lat_ms=inf_lat, post_lat_ms=post_lat,
                pre_e_mJ=pre_energy, inf_e_mJ=inf_energy, post_e_mJ=post_energy,
                pre_max_v=pre_max_v, pre_mean_v=pre_mean_v, pre_max_c=pre_max_c, pre_mean_c=pre_mean_c,
                inf_max_v=inf_max_v, inf_mean_v=inf_mean_v, inf_max_c=inf_max_c, inf_mean_c=inf_mean_c,
                post_max_v=post_max_v, post_mean_v=post_mean_v, post_max_c=post_max_c, post_mean_c=post_mean_c,
                pre_pwr=pre_pwr, inf_pwr=inf_pwr, post_pwr=post_pwr,
                memory=memory
            )

            t.sleep(1)

# === MENU ===
def menu():
    print("\n--- Select Mode ---\n1: CPU 1 Thread\n2: CPU 4 Threads\n3: CPU 8 Threads\n4: Quit\n")
    choice = input("Enter your choice: ").strip()
    if choice=='1':
        interpreter = load_model(num_threads=1)
        run_inference(interpreter, labels, mode="CPU1")
    elif choice=='2':
        interpreter = load_model(num_threads=4)
        run_inference(interpreter, labels, mode="CPU4")
    elif choice=='3':
        interpreter = load_model(num_threads=0)
        run_inference(interpreter, labels, mode="GPU")
    elif choice=='4':
        exit()
    else:
        print("Invalid choice, try again.")
        menu()

# === MAIN ===
def main():
    global labels
    init_csv()
    labels = load_labels(LABEL_MAP)
    menu()

if __name__=="__main__":
    main()

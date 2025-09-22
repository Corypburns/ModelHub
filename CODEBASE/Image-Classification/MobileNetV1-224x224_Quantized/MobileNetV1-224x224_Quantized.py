from datetime import datetime as dt
from pathlib import Path
import tensorflow as tf
import time as t
import socket
import cv2, numpy as np
from jtop import jtop

host = socket.gethostname()

match host:
    case "DESKTOP-FI8GT7F":
        BASE_PATH = Path(r"E:\Code\Python\ModelHub")
        TEST_IMAGE_BASE_PATH = Path(r"E:\Code\Python\DATASETS\COCO")
    case "CoryPC":
        BASE_PATH = None # Placeholder value for my laptop
        TEST_IMAGE_BASE_PATH = None # Placeholder value for my laptop
    case "nano1-desktop": # Enter the host name of the computer you are working on
        BASE_PATH = Path("/home/nano1/anik-lab/ModelHub")
        TEST_IMAGE_BASE_PATH = Path("/home/nano1/anik-lab/coco")
    
TEST_IMAGE_PATH = TEST_IMAGE_BASE_PATH / "test2017"
MODEL_PATH = BASE_PATH / "MODELBASE" / "Image-Classification" / "MobileNetV1-224x224_Quantized" / "mobilenet_v1_1_224_quant.tflite"
LOG_DIR = BASE_PATH / "OUTPUTS" / "Image-Classification" / "MobileNetV1-224x224_Quantized"
LABEL_MAP = BASE_PATH / "LABELMAPS" / "Image-Classification" / "labels.txt"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_MNV1_224x224(quantized)_{DATE_TIME}.csv"
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

def init_csv():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'w') as f:
            f.write(HEADERS)

def append_csv_row(
        timestamp, review, mode,
        pre_lat_ms, inf_lat_ms, post_lat_ms,
        pre_e_mJ, inf_e_mJ, post_e_mJ,
        pre_max_v, pre_mean_v, pre_max_c, pre_mean_c,
        inf_max_v, inf_mean_v, inf_max_c, inf_mean_c,
        post_max_v, post_mean_v, post_max_c, post_mean_c,
        pre_pwr, inf_pwr, post_pwr      
):
    row = ",".join([
        timestamp,
        review,
        mode,
        f"{pre_lat_ms}", f"{inf_lat_ms}", f"{post_lat_ms}",
        f"{pre_e_mJ}", f"{inf_e_mJ}", f"{post_e_mJ}",
        f"{pre_max_v}", f"{pre_mean_v}", f"{pre_max_c}", f"{pre_mean_c}",
        f"{inf_max_v}", f"{inf_mean_v}", f"{inf_max_c}", f"{inf_mean_c}",
        f"{post_max_v}", f"{post_mean_v}", f"{post_max_c}", f"{post_mean_c}",
        f"{pre_pwr}", f"{inf_pwr}", f"{post_pwr}"
    ]) + "\n"
    with open(OUTPUT_PATH, 'a') as f:
        f.write(row)

def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]
    
def load_model(num_threads):
    start_load = t.time()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=num_threads)
    end_load = t.time()
    
    start_allocation = t.time()
    interpreter.allocate_tensors()
    end_allocation = t.time()
    print(f"Model load time: {(end_load - start_load) * 1000:.2f} ms.",
          f"\nModel allocation time: {(end_allocation - start_allocation) * 1000:.2f} ms.", 
          "\n\nPress 'enter' to continue...")
    input()
    
    return interpreter

# --- Jetson Stats Functions ---

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

# === PROCESS IMAGES ===
def image_processing_inference(interpreter, img_path, labels=None, mode="CPU1"):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details[0]['dtype'])
    print(input_details[0]['shape'])
    print("Quantization:", input_details[0]['quantization'])

    with jtop() as jetson:
        for img_path in TEST_IMAGE_PATH.glob("*.jpg"):
            # === Pre-Inference Stats ===
            pre_max_v, pre_mean_v, pre_max_c, pre_mean_c, pre_pwr = get_jetson_stats(jetson)
            pre_time = t.time()

            raw_img = cv2.imread(str(img_path))
            height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]
            resized_img = cv2.resize(raw_img, (width, height))
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            normalized_img = rgb_img.astype(np.uint8)
            input_tensor = np.expand_dims(normalized_img, axis=0)

            # === Inference Stats Start ===
            inf_max_v_start, inf_mean_v_start, inf_max_c_start, inf_mean_c_start, inf_pwr_start = get_jetson_stats(jetson)
            inf_start = t.time()

            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()

            inf_end = t.time()
            inf_max_v_end, inf_mean_v_end, inf_max_c_end, inf_mean_c_end, inf_pwr_end = get_jetson_stats(jetson)

            post_time = t.time()
            post_max_v, post_mean_v, post_max_c, post_mean_c, post_pwr = get_jetson_stats(jetson)

            output = interpreter.get_tensor(output_details[0]['index'])[0]
            predicted_index = output.argmax()
            predicted_label = labels[predicted_index]
            confidence = output[predicted_index]

            # Compute stats
            pre_lat = (inf_start - pre_time) * 1000
            inf_lat = (inf_end - inf_start) * 1000
            post_lat = (post_time - inf_end) * 1000

            # Energy = Power (mW) * time (s) = mJ
            pre_energy = pre_pwr * (inf_start - pre_time)
            inf_energy = ((inf_pwr_start + inf_pwr_end) / 2) * (inf_end - inf_start)
            post_energy = post_pwr * (post_time - inf_end)

            # Average voltage/current during inference
            inf_max_v = max(inf_max_v_start, inf_max_v_end)
            inf_mean_v = (inf_mean_v_start + inf_mean_v_end) / 2
            inf_max_c = max(inf_max_c_start, inf_max_c_end)
            inf_mean_c = (inf_mean_c_start + inf_mean_c_end) / 2

            print(f"Image: {img_path.name} | ({inf_lat:.2f} ms.)")

            print(f"""
--- Inference Stats ---
Image: {img_path.name}
Prediction: {predicted_label} ({confidence * 100:.2f}%)

 Latencies (ms):
    Pre-processing : {pre_lat}
    Inference      : {inf_lat}
    Post-processing: {post_lat}

 Energy (mJ):
    Pre-processing : {pre_energy}
    Inference      : {inf_energy}
    Post-processing: {post_energy}

 Power (mW):
    Pre-processing : {pre_pwr}
    Inference      : {(inf_pwr_start + inf_pwr_end)/2}
    Post-processing: {post_pwr}

 Voltage (V):
    Pre-processing : {pre_max_v}
    Inference      : {inf_mean_v}
    Post-processing: {post_max_v}

 Current (A):
    Pre-processing : {pre_max_c}
    Inference      : {inf_mean_c}
    Post-processing: {post_max_c}
""")

            t.sleep(1)

            # === Log Results ===
            append_csv_row(
                timestamp=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                review=img_path.name,
                mode=mode,
                pre_lat_ms=pre_lat,
                inf_lat_ms=inf_lat,
                post_lat_ms=post_lat,
                pre_e_mJ=pre_energy,
                inf_e_mJ=inf_energy,
                post_e_mJ=post_energy,
                pre_max_v=pre_max_v, pre_mean_v=pre_mean_v, pre_max_c=pre_max_c, pre_mean_c=pre_mean_c,
                inf_max_v=inf_max_v, inf_mean_v=inf_mean_v, inf_max_c=inf_max_c, inf_mean_c=inf_mean_c,
                post_max_v=post_max_v, post_mean_v=post_mean_v, post_max_c=post_max_c, post_mean_c=post_mean_c,
                pre_pwr=pre_pwr, inf_pwr=(inf_pwr_start + inf_pwr_end) / 2, post_pwr=post_pwr
            )

def menu():
    print("\n--- Select Mode ---\n1: CPU 1 Thread\n2: CPU 4 Threads\n3: CPU 8 Threads\n4: Quit\n")
    choice = input("Enter your choice: ").strip()
    if choice == '1':
        interpreter = load_model(num_threads=1)
        image_processing_inference(interpreter, TEST_IMAGE_PATH, labels, mode="CPU1")
    elif choice == '2':
        interpreter = load_model(num_threads=4)
        image_processing_inference(interpreter, TEST_IMAGE_PATH, labels, mode="CPU4")
    elif choice == '3':
        interpreter = load_model(num_threads=8)
        image_processing_inference(interpreter, TEST_IMAGE_PATH, labels, mode="CPU8")
    elif choice == '4':
        print("Exiting...")
        exit()
    else:
        print("Invalid choice, try again.")
        menu()

def main():
    global labels
    init_csv()
    labels = load_labels(LABEL_MAP)
    menu()

if __name__ == "__main__":
    main()


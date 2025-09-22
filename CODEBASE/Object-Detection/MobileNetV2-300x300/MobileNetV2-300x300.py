import tensorflow as tf
import time as t
from datetime import datetime as dt
from pathlib import Path
import socket
import cv2
import numpy as np
from jtop import jtop

# === CONFIG ===
host = socket.gethostname()
match host:
    case "DESKTOP-FI8GT7F":
        BASE_PATH = Path(r"E:\Code\Python\ModelHub")
        TEST_IMAGE_BASE_PATH = Path(r"E:\Code\Python\DATASETS\COCO")
    case "CoryPC":
        BASE_PATH = None  # Placeholder for your laptop
        TEST_IMAGE_BASE_PATH = None
    case "nano1-desktop":
        BASE_PATH = Path("/home/nano1/anik-lab/ModelHub")
        TEST_IMAGE_BASE_PATH = Path("/home/nano1/anik-lab/coco")

TEST_IMAGE_PATH = TEST_IMAGE_BASE_PATH / "test2017"
MODEL_PATH = BASE_PATH / "MODELBASE" / "Object-Detection" / "MobileNetV2-300x300" / "ssd_mobilenet_v2_300x300.tflite"
LABEL_MAP = BASE_PATH / "LABELMAPS" / "Object-Detection" / "labelmap.txt"
LOG_DIR = BASE_PATH / "OUTPUTS" / "Object-Detection" / "MobileNetV2-300x300"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_MNV2_300x300(unquantized)_{DATE_TIME}.csv"
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

# === CSV METHODS ===
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
        f"{pre_lat_ms:.1f}", f"{inf_lat_ms:.1f}", f"{post_lat_ms:.1f}",
        f"{pre_e_mJ:.1f}", f"{inf_e_mJ:.1f}", f"{post_e_mJ:.1f}",
        f"{pre_max_v:.2f}", f"{pre_mean_v:.2f}", f"{pre_max_c:.2f}", f"{pre_mean_c:.2f}",
        f"{inf_max_v:.2f}", f"{inf_mean_v:.2f}", f"{inf_max_c:.2f}", f"{inf_mean_c:.2f}",
        f"{post_max_v:.2f}", f"{post_mean_v:.2f}", f"{post_max_c:.2f}", f"{post_mean_c:.2f}",
        f"{pre_pwr:.2f}", f"{inf_pwr:.2f}", f"{post_pwr:.2f}"
    ]) + "\n"
    with open(OUTPUT_PATH, 'a') as f:
        f.write(row)

# === LOAD LABEL MAP ===
def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# === LOAD MODEL ===
def load_model(num_threads=1):
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

# === DRAW BOXES ===
def draw_boxes(image, boxes, classes, scores, num_detections, labels=None):
    h, w, _ = image.shape
    threshold = 0.5

    for i in range(int(num_detections)):
        if scores[i] < threshold:
            continue

        y_min, x_min, y_max, x_max = boxes[i]
        x_min = int(x_min * w)
        x_max = int(x_max * w)
        y_min = int(y_min * h)
        y_max = int(y_max * h)

        class_id = int(classes[i])
        confidence = scores[i]

        label = f"{labels[class_id]}" if labels and class_id < len(labels) else f"ID {class_id}"
        label_text = f"{label}: {confidence:.2f}"

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, label_text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# === PROCESS IMAGES ===
def image_processing_inference(interpreter, labels=None, mode="CPU1"):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Quantization:", input_details[0]['quantization'])

    with jtop() as jetson:
        for img_path in TEST_IMAGE_PATH.glob("*.jpg"):
            # === Pre-Inference Stats ===
            pre_max_v, pre_mean_v, pre_max_c, pre_mean_c, pre_pwr = get_jetson_stats(jetson)
            pre_start = t.time()

            img_raw = cv2.imread(str(img_path))
            if img_raw is None:
                print(f"Image not found: {img_path}")
                continue

            img_resized = cv2.resize(img_raw, (300, 300))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_input = img_rgb.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(img_input, axis=0)

            # === Inference Start Stats ===
            inf_max_v_start, inf_mean_v_start, inf_max_c_start, inf_mean_c_start, inf_pwr_start = get_jetson_stats(jetson)
            inf_start = t.time()

            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()

            inf_end = t.time()
            inf_max_v_end, inf_mean_v_end, inf_max_c_end, inf_mean_c_end, inf_pwr_end = get_jetson_stats(jetson)

            # === Post-Inference Stats ===
            post_time = t.time()
            post_max_v, post_mean_v, post_max_c, post_mean_c, post_pwr = get_jetson_stats(jetson)

            # Outputs
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
            num_detections = interpreter.get_tensor(output_details[3]['index'])[0]

            drawn = draw_boxes(img_raw.copy(), boxes, classes, scores, num_detections, labels)
            cv2.imshow("Detections", drawn)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            predicted_label = "N/A"
            confidence = 0.0
            if labels:
                max_idx = np.argmax(scores)
                if scores[max_idx] > 0.5 and max_idx < len(labels):
                    predicted_label = labels[int(classes[max_idx])]
                    confidence = scores[max_idx]

            # Compute latencies
            pre_lat = (inf_start - pre_start) * 1000
            inf_lat = (inf_end - inf_start) * 1000
            post_lat = (post_time - inf_end) * 1000

            # Energy = Power (mW) * time (s) = mJ
            pre_energy = pre_pwr * (inf_start - pre_start)
            inf_energy = ((inf_pwr_start + inf_pwr_end) / 2) * (inf_end - inf_start)
            post_energy = post_pwr * (post_time - inf_end)

            # Average voltage/current during inference
            inf_max_v = max(inf_max_v_start, inf_max_v_end)
            inf_mean_v = (inf_mean_v_start + inf_mean_v_end) / 2
            inf_max_c = max(inf_max_c_start, inf_max_c_end)
            inf_mean_c = (inf_mean_c_start + inf_mean_c_end) / 2

            timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"Image: {img_path.name} | ({inf_lat:.2f} ms.)")

            print(f"""
--- Inference Stats ---
Image: {img_path.name}
Prediction: {predicted_label} ({confidence * 100:.2f}%)

 Latencies (ms):
    Pre-processing : {pre_lat:.2f}
    Inference      : {inf_lat:.2f}
    Post-processing: {post_lat:.2f}

 Energy (mJ):
    Pre-processing : {pre_energy:.2f}
    Inference      : {inf_energy:.2f}
    Post-processing: {post_energy:.2f}

 Power (mW):
    Pre-processing : {pre_pwr:.2f}
    Inference      : {(inf_pwr_start + inf_pwr_end)/2:.2f}
    Post-processing: {post_pwr:.2f}

 Voltage (V):
    Pre-processing : {pre_max_v:.2f}
    Inference      : {inf_mean_v:.2f}
    Post-processing: {post_max_v:.2f}

 Current (A):
    Pre-processing : {pre_max_c:.2f}
    Inference      : {inf_mean_c:.2f}
    Post-processing: {post_max_c:.2f}
""")

            t.sleep(1)  # Pause between images

            # Log results
            append_csv_row(
                timestamp=timestamp,
                review=img_path.name,
                mode=mode,
                pre_lat_ms=pre_lat,
                inf_lat_ms=inf_lat,
                post_lat_ms=post_lat,
                pre_e_mJ=pre_energy,
                inf_e_mJ=inf_energy,
                post_e_mJ=post_energy,
                pre_max_v=pre_max_v,
                pre_mean_v=pre_mean_v,
                pre_max_c=pre_max_c,
                pre_mean_c=pre_mean_c,
                inf_max_v=inf_max_v,
                inf_mean_v=inf_mean_v,
                inf_max_c=inf_max_c,
                inf_mean_c=inf_mean_c,
                post_max_v=post_max_v,
                post_mean_v=post_mean_v,
                post_max_c=post_max_c,
                post_mean_c=post_mean_c,
                pre_pwr=pre_pwr,
                inf_pwr=(inf_pwr_start + inf_pwr_end)/2,
                post_pwr=post_pwr
            )
            
         
def menu():
    print("\n--- Select Mode ---\n1: CPU 1 Thread\n2: CPU 4 Threads\n3: GPU\n4: Quit\n")
    choice = input("Enter your choice: ").strip()

    if choice == '1':
        interpreter = load_model(num_threads=1)
        init_csv()
        labels = load_labels(LABEL_MAP)
        image_processing_inference(interpreter, labels, mode="CPU1")

    elif choice == '2':
        interpreter = load_model(num_threads=4)
        init_csv()
        labels = load_labels(LABEL_MAP)
        image_processing_inference(interpreter, labels, mode="CPU4")

    elif choice == '3':
        interpreter = load_model(num_threads=0)
        init_csv()
        labels = load_labels(LABEL_MAP)
        image_processing_inference(interpreter, labels, mode="GPU")

    elif choice == '4':
        print("Exiting...")
        exit()

    else:
        print("Invalid choice, try again.")
        menu()


if __name__ == "__main__":
    menu()

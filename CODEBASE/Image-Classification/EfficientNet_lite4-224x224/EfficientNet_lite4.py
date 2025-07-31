import tensorflow as tf
import time as t
from datetime import datetime as dt
import keyboard as k
from pathlib import Path
import numpy as np
import socket
import cv2
from jtop import jtop

host = socket.gethostname()
print(host)

# === CONFIG ===
match host:
    case "CoryPC":
        BASE_PATH = Path(r"C:\Users\burns\OneDrive\Desktop\Projects\Python\ModelHub")
        TEST_IMAGE_BASE_PATH = Path(r"C:\Users\burns\OneDrive\Desktop\Datasets")
    case "Place_holder":
        BASE_PATH = Path(r"E:\Code\Python\ModelHub")
        TEST_IMAGE_BASE_PATH = Path(r"C:\Users\burns\OneDrive\Desktop\Datasets")
    case "nano1-desktop":
        BASE_PATH = Path.home() / "nano1" / "anik-lab" / "ModelHub"
        TEST_IMAGE_BASE_PATH = Path.home() / "nano1" / "anik-lab" / "coco"

TEST_IMAGE_PATH = TEST_IMAGE_BASE_PATH / "test2017"
MODEL_PATH = BASE_PATH / "MODELBASE" / "Image-Classification" / "EfficientNet_lite4-224x224" / "efficientnet_lite4_fp32_2.tflite"
LABEL_MAP = BASE_PATH / "LABELMAPS" / "Image-Classification" / "labels.txt"
LOG_DIR = BASE_PATH / "OUTPUTS" / "Image-Classification" / "EfficientNet_lite4-224x224"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_ENL4_224x224(float)_{DATE_TIME}.csv"
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

def append_csv_row(*args):
    row = ",".join(map(str, args)) + "\n"
    with open(OUTPUT_PATH, 'a') as f:
        f.write(row)

# === LOAD LABEL MAP ===
def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# === LOAD MODEL WITH MODE ===
def load_model(mode="CPU-1"):
    start_load = t.time()
    interpreter = None

    if mode.startswith("CPU"):
        num_threads = int(mode.split("-")[1])
        print(f"[INFO] Using CPU with {num_threads} thread(s)")
        interpreter = tf.lite.Interpreter(
            model_path=str(MODEL_PATH),
            num_threads=num_threads
        )

    elif mode == "GPU":
        try:
            delegate = tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so")
            interpreter = tf.lite.Interpreter(
                model_path=str(MODEL_PATH),
                experimental_delegates=[delegate]
            )
            print("[INFO] Using GPU delegate")
        except Exception as e:
            print(f"[ERROR] GPU delegate failed: {e}")
            print("[INFO] Falling back to CPU with 1 thread")
            interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=1)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    end_load = t.time()
    start_alloc = t.time()
    interpreter.allocate_tensors()
    end_alloc = t.time()

    print(f"Model load time: {(end_load - start_load)*1000:.2f} ms")
    print(f"Model allocation time: {(end_alloc - start_alloc)*1000:.2f} ms")
    print("Press SPACE to continue...")
    k.wait('space')

    return interpreter

# === SAMPLE POWER/VOLTAGE/CURRENT USING JTOP ===
def sample_stats(jetson, duration=0.5, sampling_interval=0.05):
    voltages = []
    currents = []
    powers = []
    samples = int(duration / sampling_interval)

    for _ in range(samples):
        stats = jetson.stats
        # Use 'VDD_IN' rail as example - change as needed
        voltage = stats['voltages'].get('VDD_IN', 0)
        current = stats['currents'].get('VDD_IN', 0)
        power = stats['power'].get('VDD_IN', 0)

        voltages.append(voltage)
        currents.append(current)
        powers.append(power)

        t.sleep(sampling_interval)

    mean_V = np.mean(voltages)
    max_V = np.max(voltages)
    mean_C = np.mean(currents)
    max_C = np.max(currents)
    mean_P = np.mean(powers)
    max_P = np.max(powers)

    energy_mJ = mean_P * duration * 1000  # power(W) * seconds * 1000 = mJ
    power_mW = mean_P * 1000

    return {
        'mean_V': mean_V,
        'max_V': max_V,
        'mean_C': mean_C,
        'max_C': max_C,
        'power_mW': power_mW,
        'energy_mJ': energy_mJ,
    }

# === IMAGE PROCESSING WITH JTOP POWER LOGGING ===
def image_processing(interpreter, labels, input_shape, mode):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with jtop() as jetson:
        if not jetson.ok():
            print("jtop service not running properly. Exiting.")
            return

        for img_path in TEST_IMAGE_PATH.glob("*.jpg"):

            # Pre-inference sample
            pre_stats = sample_stats(jetson, duration=0.5)

            # Load & preprocess image
            img_raw = cv2.imread(str(img_path))
            if img_raw is None:
                print(f"Image not found or unreadable: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (input_shape[2], input_shape[1]))
            img_normalized = img_resized.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(img_normalized, axis=0)

            # Inference
            inf_start = t.time()
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            inf_end = t.time()

            # Post-inference sample
            post_stats = sample_stats(jetson, duration=0.5)

            # Output processing
            output = interpreter.get_tensor(output_details[0]['index'])
            top_idx = np.argmax(output[0])
            pred_label = labels[top_idx]
            confidence = output[0][top_idx]

            # Timing calculations (ms)
            pre_latency = 500  # fixed sample time (0.5s)
            inf_latency = (inf_end - inf_start) * 1000
            post_latency = 500  # fixed sample time (0.5s)

            # Append CSV row
            append_csv_row(
                dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                img_path.name,
                mode,
                f"{pre_latency:.2f}", f"{inf_latency:.2f}", f"{post_latency:.2f}",
                f"{pre_stats['energy_mJ']:.3f}", "", f"{post_stats['energy_mJ']:.3f}",
                f"{pre_stats['max_V']:.3f}", f"{pre_stats['mean_V']:.3f}",
                f"{pre_stats['max_C']:.3f}", f"{pre_stats['mean_C']:.3f}",
                "", "", "", "",  # placeholders for inf current/volt (add if you want)
                f"{post_stats['max_V']:.3f}", f"{post_stats['mean_V']:.3f}",
                f"{post_stats['max_C']:.3f}", f"{post_stats['mean_C']:.3f}",
                f"{pre_stats['power_mW']:.3f}", "", f"{post_stats['power_mW']:.3f}"
            )

            print(f"{img_path.name} → {pred_label} ({confidence*100:.2f}%), Inference Time: {inf_latency:.2f} ms")
            print("Press SPACE to continue...")
            k.wait('space')

# === MENU ===
def menu():
    print("Choose inference mode:")
    print("1) CPU (1 thread)")
    print("2) CPU (4 threads)")
    print("3) GPU (if supported)")
    choice = input("Enter choice (1-3): ")
    if choice == "1":
        return "CPU-1"
    elif choice == "2":
        return "CPU-4"
    elif choice == "3":
        return "GPU"
    else:
        print("Invalid choice. Defaulting to CPU-1")
        return "CPU-1"

def main():
    mode = menu()
    interpreter = load_model(mode=mode)
    labels = load_labels(LABEL_MAP)
    input_shape = interpreter.get_input_details()[0]['shape']
    init_csv()
    image_processing(interpreter, labels, input_shape, mode)

if __name__ == "__main__":
    main()

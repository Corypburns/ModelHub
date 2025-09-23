import tensorflow as tf
import matplotlib.pyplot as mpl
import cv2
import time as t
from datetime import datetime as dt
from pathlib import Path
import socket
import numpy as np
from jtop import jtop   # Jetson stats

# === CONFIG ===
host = socket.gethostname()
print("Running on host:", host)

match host:
    case "DESKTOP-FI8GT7F":
        BASE_PATH = Path(r"E:\Code\Python\ModelHub")
    case "archlinux":
        BASE_PATH = Path("/home/cburns/Desktop/ModelHub")
    case "nano1-desktop":
        BASE_PATH = Path("/home/nano1/anik-lab/ModelHub")

IMAGE_PATH = BASE_PATH / "DATASETS" / "Super-Resolution"
MODEL_PATH = BASE_PATH / "MODELBASE" / "Super-Resolution" / "ESRGAN.tflite"
LOG_DIR = BASE_PATH / "OUTPUTS" / "Super-Resolution"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_esrgantf_{DATE_TIME}.csv"
OUTPUT_PATH = LOG_DIR / FILE_NAME

HEADERS = (
    "Timestamp,Review,Mode,"
    "Pre_Lat_ms,Inf_Lat_ms,Post_Lat_ms,"
    "Pre_E_mJ,Inf_E_mJ,Post_E_mJ,"
    "Pre_Max_V,Pre_Mean_V,Pre_Max_C,Pre_Mean_C,"
    "Inf_Max_V,Inf_Mean_V,Inf_Max_C,Inf_Mean_C,"
    "Post_Max_V,Post_Mean_V,Post_Max_C,Post_Mean_C,"
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW,Mem_Used_MB\n"
)

# === CSV FUNCTIONS ===
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

# === LOAD MODEL ===
def load_model(num_threads):
    start_load = t.time()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=num_threads)
    end_load = t.time()

    start_alloc = t.time()
    interpreter.allocate_tensors()
    end_alloc = t.time()

    print(
        f"Model Load Time: {(end_load - start_load)*1000:.2f} ms",
        f"\nModel Allocation Time: {(end_alloc - start_alloc)*1000:.2f} ms",
        "\n\nPress Enter to continue..."
    )
    input()

    return interpreter

# === PROCESSING & INFERENCE ===
def run_inference(interpreter, img_path, mode="CPU1"):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with jtop() as jetson:
        # === PREPROCESS ===
        pre_max_v, pre_mean_v, pre_max_c, pre_mean_c, pre_pwr = get_jetson_stats(jetson)
        pre_start = t.time()

        lr = cv2.imread(str(img_path))
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        lr = tf.expand_dims(lr, axis=0)
        lr = tf.cast(lr, tf.float32)

        pre_end = t.time()

        # === INFERENCE ===
        inf_max_v_start, inf_mean_v_start, inf_max_c_start, inf_mean_c_start, inf_pwr_start = get_jetson_stats(jetson)
        inf_start = t.time()
        interpreter.set_tensor(input_details[0]['index'], lr)
        interpreter.invoke()
        inf_end = t.time()
        inf_max_v_end, inf_mean_v_end, inf_max_c_end, inf_mean_c_end, inf_pwr_end = get_jetson_stats(jetson)

        # === POSTPROCESS ===
        post_start = t.time()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        sr = tf.squeeze(output_data, axis=0)
        sr = tf.clip_by_value(sr, 0, 255)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint8)
        post_end = t.time()
        post_max_v, post_mean_v, post_max_c, post_mean_c, post_pwr = get_jetson_stats(jetson)

        # Memory stats
        stats = jetson.stats
        ram_stats = stats["RAM"]
        if isinstance(ram_stats, dict):
            ram_used = float(ram_stats.get("used", 0))
        else:
            ram_used = float(ram_stats)

    # === DISPLAY ===
    lr_disp = tf.cast(tf.squeeze(lr, axis=0), tf.uint8)

    mpl.figure(figsize=(10, 4))
    mpl.subplot(1, 2, 1)
    mpl.title("ESRGAN (x4)")
    mpl.imshow(sr.numpy())

    bicubic = tf.image.resize(lr_disp, [200, 200], tf.image.ResizeMethod.BICUBIC)
    bicubic = tf.cast(bicubic, tf.uint8)

    mpl.subplot(1, 2, 2)
    mpl.title("Bicubic")
    mpl.imshow(bicubic.numpy())
    mpl.show()

    # === Latency ===
    pre_lat = (pre_end - pre_start) * 1000
    inf_lat = (inf_end - inf_start) * 1000
    post_lat = (post_end - post_start) * 1000

    # === Energy ===
    pre_energy = pre_pwr * (inf_start - pre_start)
    inf_energy = ((inf_pwr_start + inf_pwr_end) / 2) * (inf_end - inf_start)
    post_energy = post_pwr * (post_end - inf_end)

    # === Inference averaged stats ===
    inf_max_v = max(inf_max_v_start, inf_max_v_end)
    inf_mean_v = (inf_mean_v_start + inf_mean_v_end) / 2
    inf_max_c = max(inf_max_c_start, inf_max_c_end)
    inf_mean_c = (inf_mean_c_start + inf_mean_c_end) / 2
    inf_pwr = (inf_pwr_start + inf_pwr_end) / 2

    # === LOG ===
    append_csv_row(
        timestamp=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        review=img_path.name,
        mode=mode,
        pre_lat_ms=pre_lat, inf_lat_ms=inf_lat, post_lat_ms=post_lat,
        pre_e_mJ=pre_energy, inf_e_mJ=inf_energy, post_e_mJ=post_energy,
        pre_max_v=pre_max_v, pre_mean_v=pre_mean_v, pre_max_c=pre_max_c, pre_mean_c=pre_mean_c,
        inf_max_v=inf_max_v, inf_mean_v=inf_mean_v, inf_max_c=inf_max_c, inf_mean_c=inf_mean_c,
        post_max_v=post_max_v, post_mean_v=post_mean_v, post_max_c=post_max_c, post_mean_c=post_mean_c,
        pre_pwr=pre_pwr, inf_pwr=inf_pwr, post_pwr=post_pwr, memory=ram_used
    )

    # Print summary
    print(f"\n--- Inference Stats for {img_path.name} ---")
    print(f" Latencies (ms): pre={pre_lat:.2f}, inf={inf_lat:.2f}, post={post_lat:.2f}")
    print(f" Energy (mJ): pre={pre_energy:.2f}, inf={inf_energy:.2f}, post={post_energy:.2f}")
    print(f" Power (mW): pre={pre_pwr:.2f}, inf={inf_pwr:.2f}, post={post_pwr:.2f}")
    print(f" Voltage (V): pre={pre_max_v:.2f}, inf={inf_mean_v:.2f}, post={post_max_v:.2f}")
    print(f" Current (A): pre={pre_max_c:.2f}, inf={inf_mean_c:.2f}, post={post_max_c:.2f}")
    print(f" Memory Used (MB): {ram_used:.2f}")

    t.sleep(1)

# === MAIN MENU ===
def menu():
    print("Super-Resolution Inference Mode\n1) CPU1\n2) CPU4\n3) GPU (NOT IMPLEMENTED)\n")
    choice = int(input("-> "))

    match choice:
        case 1:
            mode = "CPU1"
            interpreter = load_model(num_threads=1)
        case 2:
            mode = "CPU4"
            interpreter = load_model(num_threads=4)
        case 3:
            print("GPU not supported in TFLite on this script.")
            return
        case _:
            print("Invalid choice.")
            return

    for img_path in IMAGE_PATH.glob("*.jpg"):
        run_inference(interpreter, img_path, mode)

# === DRIVER FUNCTION ===
def main():
    init_csv()
    menu()

if __name__ == "__main__":
    main()

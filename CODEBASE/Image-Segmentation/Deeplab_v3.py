import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime as dt
import time as t
import os
from jtop import jtop
import matplotlib.pyplot as plt

# === CONFIG ===
VOC_DIR = Path("/home/nano1/anik-lab/VOC2012/VOC2012_train_val")
MODEL_PATH = Path("/home/nano1/anik-lab/ModelHub/MODELBASE/Image-Segmentation/deeplab_v3.tflite")
LOG_DIR = Path("/home/nano1/anik-lab/ModelHub/OUTPUTS/Image-Segmentation/Deeplab_v3")
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_deeplabv3_{DATE_TIME}.csv"
OUTPUT_PATH = LOG_DIR / FILE_NAME

HEADERS = (
    "Timestamp,ImageID,Mode,"
    "Pre_Lat_ms,Inf_Lat_ms,Post_Lat_ms,"
    "Pre_E_mJ,Inf_E_mJ,Post_E_mJ,"
    "Pre_Max_V,Pre_Mean_V,Pre_Max_C,Pre_Mean_C,"
    "Inf_Max_V,Inf_Mean_V,Inf_Max_C,Inf_Mean_C,"
    "Post_Max_V,Post_Mean_V,Post_Max_C,Post_Mean_C,"
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW\n"
)

# === CSV FUNCTIONS ===
def init_csv():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'w') as f:
            f.write(HEADERS)

def append_csv_row(timestamp, image_id, mode,
                   pre_lat_ms, inf_lat_ms, post_lat_ms,
                   pre_e_mJ, inf_e_mJ, post_e_mJ,
                   pre_max_v, pre_mean_v, pre_max_c, pre_mean_c,
                   inf_max_v, inf_mean_v, inf_max_c, inf_mean_c,
                   post_max_v, post_mean_v, post_max_c, post_mean_c,
                   pre_pwr, inf_pwr, post_pwr):
    row = ",".join(map(str, [
        timestamp, image_id, mode,
        f"{pre_lat_ms:.4f}", f"{inf_lat_ms:.4f}", f"{post_lat_ms:.4f}",
        f"{pre_e_mJ:.4f}", f"{inf_e_mJ:.4f}", f"{post_e_mJ:.4f}",
        f"{pre_max_v:.4f}", f"{pre_mean_v:.4f}", f"{pre_max_c:.4f}", f"{pre_mean_c:.4f}",
        f"{inf_max_v:.4f}", f"{inf_mean_v:.4f}", f"{inf_max_c:.4f}", f"{inf_mean_c:.4f}",
        f"{post_max_v:.4f}", f"{post_mean_v:.4f}", f"{post_max_c:.4f}", f"{post_mean_c:.4f}",
        f"{pre_pwr:.4f}", f"{inf_pwr:.4f}", f"{post_pwr:.4f}"
    ])) + "\n"
    with open(OUTPUT_PATH, 'a') as f:
        f.write(row)

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

# === IMAGE PREPROCESSING ===
def preprocess_image(image_path, target_size):
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize(target_size)
    array = np.array(image_resized, dtype=np.float32)
    array = array / 127.5 - 1.0
    return np.expand_dims(array, axis=0), image

def label_to_color_image(label):
    colors = np.array([
        [0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],
        [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
        [64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],
        [192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,12]
    ], dtype=np.uint8)
    return colors[label]

def get_ground_truth(mask_path, target_size):
    mask = Image.open(mask_path).resize(target_size, Image.NEAREST)
    array = np.array(mask)
    array[array==255] = 0
    return label_to_color_image(array)

def run_inference(interpreter, input_tensor):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return np.argmax(output_data, axis=-1)

def display_results(original, predicted, ground_truth, pause=3):
    plt.figure(figsize=(15,5))
    for idx, (img,title) in enumerate(zip([original, predicted, ground_truth],
                                         ["Original Image","Predicted Segmentation","Ground Truth"])):
        plt.subplot(1,3,idx+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(pause)
    plt.close()

# === PIPELINE ===
def run_deeplab_pipeline(num_threads, mode='CPU1'):
    init_csv()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=num_threads)
    interpreter.allocate_tensors()
    input_shape = interpreter.get_input_details()[0]['shape']
    h, w = input_shape[1], input_shape[2]

    val_txt = VOC_DIR / "ImageSets/Segmentation/val.txt"
    with open(val_txt) as f:
        val_ids = f.read().splitlines()

    with jtop() as jetson:
        for idx, image_id in enumerate(val_ids):
            image_path = VOC_DIR / "JPEGImages" / f"{image_id}.jpg"
            mask_path = VOC_DIR / "SegmentationClass" / f"{image_id}.png"
            if not image_path.exists() or not mask_path.exists():
                continue

            pre_max_v, pre_mean_v, pre_max_c, pre_mean_c, pre_pwr = get_jetson_stats(jetson)
            pre_time = t.time()
            input_tensor, original_image = preprocess_image(image_path, (w,h))
            pre_lat = (t.time()-pre_time)*1000
            pre_energy = pre_pwr*(pre_lat/1000)

            inf_max_v_start, inf_mean_v_start, inf_max_c_start, inf_mean_c_start, inf_pwr_start = get_jetson_stats(jetson)
            inf_start = t.time()
            pred_map = run_inference(interpreter, input_tensor)
            inf_end = t.time()
            inf_max_v_end, inf_mean_v_end, inf_max_c_end, inf_mean_c_end, inf_pwr_end = get_jetson_stats(jetson)
            inf_lat = (inf_end-inf_start)*1000
            inf_energy = ((inf_pwr_start+inf_pwr_end)/2)*(inf_lat/1000)
            inf_max_v = max(inf_max_v_start, inf_max_v_end)
            inf_mean_v = (inf_mean_v_start+inf_mean_v_end)/2
            inf_max_c = max(inf_max_c_start, inf_max_c_end)
            inf_mean_c = (inf_mean_c_start+inf_mean_c_end)/2
            inf_pwr = (inf_pwr_start+inf_pwr_end)/2

            post_time = t.time()
            predicted_color = label_to_color_image(pred_map)
            ground_truth_color = get_ground_truth(mask_path, (w,h))
            post_max_v, post_mean_v, post_max_c, post_mean_c, post_pwr = get_jetson_stats(jetson)
            post_lat = (t.time()-post_time)*1000
            post_energy = post_pwr*(post_lat/1000)

            print(f"\n--- Image {image_id} ---")
            print(f"Latencies (ms): Pre={pre_lat:.2f}, Inf={inf_lat:.2f}, Post={post_lat:.2f}")
            print(f"Energy (mJ): Pre={pre_energy:.4f}, Inf={inf_energy:.4f}, Post={post_energy:.4f}")
            print(f"Power (mW): Pre={pre_pwr:.4f}, Inf={inf_pwr:.4f}, Post={post_pwr:.4f}")
            print(f"Voltage (V): Pre={pre_max_v:.4f}, Inf={inf_mean_v:.4f}, Post={post_max_v:.4f}")
            print(f"Current (A): Pre={pre_max_c:.4f}, Inf={inf_mean_c:.4f}, Post={post_max_c:.4f}")

            display_results(original_image, predicted_color, ground_truth_color)
            t.sleep(1)

            append_csv_row(
                timestamp=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                image_id=image_id,
                mode=mode,
                pre_lat_ms=pre_lat, inf_lat_ms=inf_lat, post_lat_ms=post_lat,
                pre_e_mJ=pre_energy, inf_e_mJ=inf_energy, post_e_mJ=post_energy,
                pre_max_v=pre_max_v, pre_mean_v=pre_mean_v, pre_max_c=pre_max_c, pre_mean_c=pre_mean_c,
                inf_max_v=inf_max_v, inf_mean_v=inf_mean_v, inf_max_c=inf_max_c, inf_mean_c=inf_mean_c,
                post_max_v=post_max_v, post_mean_v=post_mean_v, post_max_c=post_max_c, post_mean_c=post_mean_c,
                pre_pwr=pre_pwr, inf_pwr=inf_pwr, post_pwr=post_pwr
            )

# === MENU ===
def menu():
    print("\n--- Select Mode ---\n1: CPU 1 Thread\n2: CPU 4 Threads\n3: GPU\n4: Quit\n")
    choice = input("Enter your choice: ").strip()
    if choice=='1':
        run_deeplab_pipeline(num_threads=1, mode='CPU1')
    elif choice=='2':
        run_deeplab_pipeline(num_threads=4, mode='CPU4')
    elif choice=='3':
        run_deeplab_pipeline(num_threads=0, mode='GPU')
    elif choice=='4':
        exit()
    else:
        print("Invalid choice, try again.")
        menu()

# === MAIN ===
if __name__=="__main__":
    init_csv()
    menu()

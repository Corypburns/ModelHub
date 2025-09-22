import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
import time as t
from jtop import jtop

# === CONFIG ===
VOC_DIR = Path("/home/nano1/anik-lab/VOC2012/VOC2012_train_val")
MODEL_PATH = Path("/home/nano1/anik-lab/ModelHub/MODELBASE/Image-Segmentation/deeplab_v3.tflite")
LOG_DIR = Path("/home/nano1/anik-lab/ModelHub/OUTPUTS/Image-Segmentation/Deeplab_v3")
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_deeplabv3_{DATE_TIME}.csv"
OUTPUT_PATH = os.path.join(LOG_DIR, FILE_NAME)
HEADERS = (
    "Timestamp,Review,Mode,"
    "Pre_Lat_ms,Inf_Lat_ms,Post_Lat_ms,"
    "Pre_E_mJ,Inf_E_mJ,Post_E_mJ,"
    "Pre_Max_V,Pre_Mean_V,Pre_Max_C,Pre_Mean_C,"
    "Inf_Max_V,Inf_Mean_V,Inf_Max_C,Inf_Mean_C,"
    "Post_Max_V,Post_Mean_V,Post_Max_C,Post_Mean_C,"
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW\n"
)

# === JETSON STATS FUNCTIONS (copied from your first code) ===
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

# === CSV UTILITY FUNCTIONS ===
def init_csv():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(OUTPUT_PATH):
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

# === IMAGE PREPROCESSING AND POSTPROCESSING ===
def label_to_color_image(label):
    colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
        [128, 192, 0], [0, 64, 12]
    ], dtype=np.uint8)
    return colors[label]

def preprocess_image(image_path, target_size):
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize(target_size)
    image_array = np.array(image_resized, dtype=np.float32)
    image_array = image_array / 127.5 - 1.0  # Normalize to [-1, 1]
    return np.expand_dims(image_array, axis=0), image

def get_ground_truth(mask_path, target_size):
    gt_mask = Image.open(mask_path).resize(target_size, resample=Image.NEAREST)
    gt_array = np.array(gt_mask)
    gt_array[gt_array == 255] = 0  # Replace ignore index with 0
    return label_to_color_image(gt_array)

def run_inference(interpreter, input_tensor):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return np.argmax(output_data, axis=-1)

def display_results(original, predicted, ground_truth, pause=3):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    for idx, (img, title) in enumerate(zip(
        [original, predicted, ground_truth],
        ["Original Image", "Predicted Segmentation", "Ground Truth Mask"]
    )):
        plt.subplot(1, 3, idx + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(pause)
    plt.close()

# === MAIN PIPELINE FUNCTION with Jetson Stats & Logging ===
def run_deeplab_v3_pipeline(num_threads, mode='CPU1'):
    print("Initializing pipeline...")
    init_csv()

    # Load model
    start_load = t.time()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=num_threads)
    end_load = t.time()
    load_total = (end_load - start_load) * 1000

    start_allocation = t.time()
    interpreter.allocate_tensors()
    end_allocation = t.time()
    allocation_total = (end_allocation - start_allocation) * 1000

    print(f"\nModel load time: {load_total:.4f} ms")
    print(f"Model allocation time: {allocation_total:.4f} ms")

    print("\nPress 'enter' to continue...")
    input()

    input_shape = interpreter.get_input_details()[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]

    val_list_path = os.path.join(VOC_DIR, "ImageSets", "Segmentation", "val.txt")
    with open(val_list_path) as f:
        val_ids = f.read().splitlines()

    with jtop() as jetson:
        for i, sample_id in enumerate(val_ids):
            print(f"[{i+1}/{len(val_ids)}] Processing: {sample_id}")

            image_path = os.path.join(VOC_DIR, "JPEGImages", f"{sample_id}.jpg")
            mask_path = os.path.join(VOC_DIR, "SegmentationClass", f"{sample_id}.png")

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                print(f"Skipping {sample_id} — missing file.")
                continue

            # === Pre-Inference Stats ===
            pre_max_v, pre_mean_v, pre_max_c, pre_mean_c, pre_pwr = get_jetson_stats(jetson)
            pre_time = t.time()

            input_tensor, original_image = preprocess_image(image_path, (input_width, input_height))

            # === Inference Stats Start ===
            inf_max_v_start, inf_mean_v_start, inf_max_c_start, inf_mean_c_start, inf_pwr_start = get_jetson_stats(jetson)
            inf_start = t.time()

            segmentation_map = run_inference(interpreter, input_tensor)

            inf_end = t.time()
            inf_max_v_end, inf_mean_v_end, inf_max_c_end, inf_mean_c_end, inf_pwr_end = get_jetson_stats(jetson)

            post_time = t.time()
            post_max_v, post_mean_v, post_max_c, post_mean_c, post_pwr = get_jetson_stats(jetson)

            predicted_color = label_to_color_image(segmentation_map)
            ground_truth_color = get_ground_truth(mask_path, (input_width, input_height))

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

            print(f"""
--- Inference Stats ---
Image: {sample_id}
 Latencies (ms):
    Pre-processing : {pre_lat:.2f}
    Inference      : {inf_lat:.2f}
    Post-processing: {post_lat:.2f}

 Energy (mJ):
    Pre-processing : {pre_energy:.4f}
    Inference      : {inf_energy:.4f}
    Post-processing: {post_energy:.4f}

 Power (mW):
    Pre-processing : {pre_pwr:.4f}
    Inference      : {(inf_pwr_start + inf_pwr_end)/2:.4f}
    Post-processing: {post_pwr:.4f}

 Voltage (V):
    Pre-processing : {pre_max_v:.4f}
    Inference      : {inf_mean_v:.4f}
    Post-processing: {post_max_v:.4f}

 Current (A):
    Pre-processing : {pre_max_c:.4f}
    Inference      : {inf_mean_c:.4f}
    Post-processing: {post_max_c:.4f}
""")

            display_results(original_image, predicted_color, ground_truth_color)

            t.sleep(1)

            # === Log Results ===
            append_csv_row(
                timestamp=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                review=sample_id,
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

    print("Pipeline complete.")

def menu():
    print("\n--- Select Mode ---\n1: CPU 1 Thread\n2: CPU 4 Threads\n3: GPU \n4: Quit\n")
    choice = input("Enter your choice: ").strip()
    if choice == '1':
        run_deeplab_v3_pipeline(num_threads=1, mode='CPU1')
    elif choice == '2':
        run_deeplab_v3_pipeline(num_threads=4, mode='CPU4')
    elif choice == '3':
        run_deeplab_v3_pipeline(num_threads=0, mode='GPU')
    elif choice == '4':
        print("Exiting...")
        exit()
    else:
        print("Invalid choice, try again.")
        menu()

# === ENTRY POINT ===
if __name__ == "__main__":
    menu()


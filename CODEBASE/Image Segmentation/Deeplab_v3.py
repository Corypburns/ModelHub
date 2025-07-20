import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
import time as t
import keyboard as k

# Need to implement model RAM usages per sample.
# Need to implement JTOP metrics as well.

# === CONFIG ===
VOC_DIR = r"E:\Code\Python\DATASETS\VOC2012_train_val"
MODEL_PATH = r"E:\Code\Python\ModelHub\MODELBASE\Image-Segmentation\deeplab_v3.tflite"
LOG_DIR = r"E:\Code\Python\ModelHub\OUTPUTS\Image-Classification\Deeplab_v3"
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

# === UTILITY FUNCTIONS ===
def init_csv():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'w') as f:
            f.write(HEADERS)

def append_csv_row(*args):
    row = ",".join(map(str, args)) + "\n"
    with open(OUTPUT_PATH, 'a') as f:
        f.write(row)

def label_to_color_image(label):
    colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
        [128, 192, 0], [0, 64, 128]
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

# === MAIN PIPELINE FUNCTION ===
def run_deeplab_v3_pipeline():
    print("Initializing pipeline...")
    init_csv()

    # Load model
    start_load = t.time()
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    end_load = t.time()
    load_total = (end_load - start_load) * 1000

    start_allocation = t.time()
    interpreter.allocate_tensors()
    end_allocation = t.time()
    allocation_total = (end_allocation - start_allocation) * 1000

    print(f"\nModel load time: {load_total:.4f} ms")
    print(f"Model allocation time: {allocation_total:.4f} ms")

    print("\nPress 'space' to continue...")
    k.wait('space')

    input_shape = interpreter.get_input_details()[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]

    # Load image IDs
    val_list_path = os.path.join(VOC_DIR, "ImageSets", "Segmentation", "val.txt")
    with open(val_list_path) as f:
        val_ids = f.read().splitlines()

    for i, sample_id in enumerate(val_ids):
        print(f"[{i+1}/{len(val_ids)}] Processing: {sample_id}")

        image_path = os.path.join(VOC_DIR, "JPEGImages", f"{sample_id}.jpg")
        mask_path = os.path.join(VOC_DIR, "SegmentationClass", f"{sample_id}.png")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Skipping {sample_id} — missing file.")
            continue

        input_tensor, original_image = preprocess_image(image_path, (input_width, input_height))
        segmentation_map = run_inference(interpreter, input_tensor)
        predicted_color = label_to_color_image(segmentation_map)
        ground_truth_color = get_ground_truth(mask_path, (input_width, input_height))

        # Display results
        display_results(original_image, predicted_color, ground_truth_color)

        # === CSV logging placeholder ===
        # append_csv_row(...) ← Fill in if energy/latency data is available

    print("Pipeline complete.")

# === ENTRY POINT ===
if __name__ == "__main__":
    run_deeplab_v3_pipeline()

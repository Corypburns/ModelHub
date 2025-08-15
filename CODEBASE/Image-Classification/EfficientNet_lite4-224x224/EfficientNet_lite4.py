from datetime import datetime as dt
from pathlib import Path
import tensorflow as tf
import time as t
import socket
import cv2, numpy as np

host = socket.gethostname()

match host:
    case "DESKTOP-FI8GT7F":
        BASE_PATH = Path(r"E:\Code\Python\ModelHub")
        TEST_IMAGE_BASE_PATH = Path(r"E:\Code\Python\DATASETS\COCO")
    case "CoryPC":
        BASE_PATH = None # Placeholder value for my laptop
        TEST_IMAGE_BASE_PATH = None # Placeholder value for my laptop
    case "Placeholder": # Enter the host name of the computer you are working on
        None
    
TEST_IMAGE_PATH = TEST_IMAGE_BASE_PATH / "test2017"
MODEL_PATH = BASE_PATH / "MODELBASE" / "Image-Classification" / "EfficientNet_lite4-224x224" / "efficientnet_lite4_fp32_2.tflite"
LOG_DIR = BASE_PATH / "OUTPUTS" / "Image-Classification" / "EfficientNet_lite4-224x224"
LABEL_MAP = BASE_PATH / "LABELMAPS" / "Image-Classification" / "labels.txt"
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

# === PROCESS IMAGES ===
def image_processing_inference(interpreter, img_path, labels=None, mode="CPU1"):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(input_details[0]['dtype'])
    print(input_details[0]['shape'])

    print("Quantization:", input_details[0]['quantization'])
    
    for img_path in TEST_IMAGE_PATH.glob("*.jpg"):
        delay_start = t.time()
        raw_img = cv2.imread(str(img_path))
        height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]
        resized_img = cv2.resize(raw_img, (width, height))
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        normalized_img = rgb_img.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized_img, axis=0)
    
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        delay_end = t.time()
        
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_index = output.argmax()
        predicted_label = labels[predicted_index]
        confidence = output[predicted_index]
        
        print(f"Image: {img_path.name} | Prediction: {predicted_label} - {(confidence) * 100:.2f}% | ({(delay_end - delay_start) * 1000:.2f} ms.)")
        t.sleep(1)
        
        append_csv_row(
            timestamp=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            review=img_path.name,
            mode=mode,
            pre_lat_ms=0,
            inf_lat_ms=0,
            post_lat_ms=0,
            pre_e_mJ=0,
            inf_e_mJ=0,
            post_e_mJ=0,
            pre_max_v=0,
            pre_mean_v=0,
            pre_max_c=0,
            pre_mean_c=0,
            inf_max_v=0,
            inf_mean_v=0,
            inf_max_c=0,
            inf_mean_c=0,
            post_max_v=0,
            post_mean_v=0,
            post_max_c=0,
            post_mean_c=0,
            pre_pwr=0,
            inf_pwr=0,
            post_pwr=0,
        )
        
def menu():
    print("Inference Mode\n1) CPU1\n2) CPU4\n3) GPU\n")
    choice = int(input("-> "))
    
    match choice:
        case 1:
            mode="CPU1"
            interpreter = load_model(num_threads=1)
            labels = load_labels(LABEL_MAP)
            image_processing_inference(interpreter, TEST_IMAGE_PATH, labels, mode)
        case 2:
            mode="CPU4"
            interpreter = load_model(num_threads=4)
            labels = load_labels(LABEL_MAP)
            image_processing_inference(interpreter, TEST_IMAGE_PATH, labels, mode)
        case 3:
            None


def main():
    init_csv()
    menu()

main()
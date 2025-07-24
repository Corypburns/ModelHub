import tensorflow as tf
import time as t
from datetime import datetime as dt
import keyboard as k
from pathlib import Path
import platform, cv2, numpy as np

# === CONFIG ===
if platform.system() == "Windows":
    BASE_PATH = Path(r"E:\Code\Python\ModelHub")
    TEST_IMAGE_BASE_PATH = Path(r"E:\Code\Python\DATASETS\COCO")
else:
    BASE_PATH = Path.home() / "ModelHub"
    TEST_IMAGE_BASE_PATH = BASE_PATH / "DATASETS" / "COCO"

TEST_IMAGE_PATH = TEST_IMAGE_BASE_PATH / "test2017"
MODEL_PATH = BASE_PATH / "MODELBASE" / "Object-Detection" / "MobileNetV2-300x300_Quantized" / "ssd_mobilenet_v2_300x300_falquan.tflite"
LABEL_MAP = BASE_PATH / "LABELMAPS" / "Object-Detection" / "labelmap.txt"
LOG_DIR = BASE_PATH / "OUTPUTS" / "Object-Detection" / "MobileNetV2-300x300_Quantized"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_MNV2_300x300(quantized)_{DATE_TIME}.csv"
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

# === LOAD MODEL ===
def load_model():
    start_load = t.time()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    end_load = t.time()

    start_allocation = t.time()
    interpreter.allocate_tensors()
    end_allocation = t.time()
    print(f"Model load time: {(end_load - start_load) * 1000:.2f} ms.",
          f"\nModel allocation time: {(end_allocation - start_allocation) * 1000:.2f} ms.", 
          "\n\nPress 'space' to continue...")
    k.wait('space')
    return interpreter

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
        cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# === PROCESS IMAGES ===
def image_processing_inference(interpreter, labels=None):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    for img_path in TEST_IMAGE_PATH.glob("*.jpg"):
        delay_start = t.time()

        img_raw = cv2.imread(str(img_path))
        if img_raw is None:
            print(f"Image not found: {img_path}")
            continue

        img_rgb = cv2.resize(img_raw, (300, 300))
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_input, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        delay_end = t.time()

        timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        append_csv_row(timestamp, 
                       img_path.name, 
                       "CPU"
        )

        print(f"Image: {img_path.name}", 
              f"Processed in: {(delay_end - delay_start) * 1000:.2f} ms")
        cv2.waitKey(1000)

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        num_detections = interpreter.get_tensor(output_details[3]['index'])[0]

        drawn = draw_boxes(img_raw.copy(), boxes, classes, scores, num_detections, labels)
        cv2.imshow("Detections", drawn)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

# === MAIN ===
def main():
    init_csv()
    interpreter = load_model()
    labels = load_labels(LABEL_MAP)
    image_processing_inference(interpreter, labels)


main()

import tensorflow as tf
import matplotlib.pyplot as mpl
import cv2, time as t
from datetime import datetime as dt
import keyboard as k
from pathlib import Path
import platform

# === CONFIG ===
if platform.system() == "Windows":
    BASE_PATH = Path(r"E:\Code\Python\ModelHub")
else:
    BASE_PATH = Path.home() / "ModelHub"

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
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW\n"
)

# === LOAD IMAGE ===
lr = cv2.imread(str(IMAGE_PATH / "lr-1.jpg"))
lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
lr = tf.expand_dims(lr, axis=0)
lr = tf.cast(lr, tf.float32)

# === UTILITY FUNCTIONS ===
def init_csv():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'w') as f:
            f.write(HEADERS)

def append_csv_row(*args):
    row = ",".join(map(str, args)) + "\n"
    with open(OUTPUT_PATH, 'a') as f:
        f.write(row)

def run_model(interpreter):
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], lr)
    interpreter.invoke()

def processing(interpreter):
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    sr = tf.squeeze(output_data, axis=0)
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.round(sr)
    sr = tf.cast(sr, tf.uint8)

    lr_disp = tf.cast(tf.squeeze(lr, axis=0), tf.uint8)

    mpl.figure(figsize=(1, 1))
    mpl.title('LR')
    mpl.imshow(lr_disp.numpy())

    mpl.figure(figsize=(10, 4))
    mpl.subplot(1, 2, 1)
    mpl.title('ESRGAN (x4)')
    mpl.imshow(sr.numpy())

    bicubic = tf.image.resize(lr_disp, [200, 200], tf.image.ResizeMethod.BICUBIC)
    bicubic = tf.cast(bicubic, tf.uint8)

    mpl.subplot(1, 2, 2)
    mpl.title('Bicubic')
    mpl.imshow(bicubic.numpy())

    mpl.show()

# === DRIVER FUNCTION ===
def main_pipeline():
    
    # === LOAD MODEL ===
    start_load = t.time()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    end_load = t.time()

    start_allocation = t.time()
    interpreter.allocate_tensors()
    end_allocation = t.time()

    print(
        f"Model Load Time: {(end_load - start_load) * 1000:.2f} ms.",
        f"\nModel Allocation Time: {(end_allocation - start_allocation) * 1000:.2f} ms.\nPress space to continue...."
    )
    k.wait('space')

    # === RUN MODEL ===
    run_model(interpreter)
    processing(interpreter)
    init_csv()

main_pipeline()

import os

import tensorflow as tf
import numpy as np
from PIL import Image
import time as t
import matplotlib.pyplot as plt
import logging

from CODEBASE.helper_functions import get_base_parser
from CODEBASE.config import *


# === CONFIG ===
VOC_DIR = DATA_SETS_PATH/ "Image-Segmentation" /'VOC_2012'/ 'VOC2012_train_val'
SEGMENTATION_MODEL_FOLDER = MODEL_BASE_PATH / "Image-Segmentation"


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
def run(mode, model, size, delay=0.5, inference_timer=None):

    mode_map = {
        'CPU1': {'threads': 1, 'str': 'CPU1'},
        'CPU4': {'threads': 4, 'str': 'CPU4'},
        'GPU':  {'threads': 0, 'str': 'GPU'}
    }
    model_path = SEGMENTATION_MODEL_FOLDER / f'{model}.tflite'

    if not os.path.isfile(model_path):
        logging.error("Model not found at: %s", model_path)
        return

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads= mode_map[mode]['threads'])
    interpreter.allocate_tensors()
    input_shape = interpreter.get_input_details()[0]['shape']
    h, w = input_shape[1], input_shape[2]

    val_txt = VOC_DIR / "ImageSets/Segmentation/val.txt"
    with open(val_txt) as f:
        val_ids = f.read().splitlines()

    if size > 0:
        val_ids = val_ids[:size]

    for idx, image_id in enumerate(val_ids):
        inference_timer.start_cycle() if inference_timer else None
        t.sleep(delay)
        image_path = VOC_DIR / "JPEGImages" / f"{image_id}.jpg"
        mask_path = VOC_DIR / "SegmentationClass" / f"{image_id}.png"
        if not image_path.exists() or not mask_path.exists():
            continue

        pre_time = t.time()
        input_tensor, original_image = preprocess_image(image_path, (w,h))
        pre_lat = (t.time()-pre_time)*1000

        if inference_timer:
            inference_timer.start_inference()
        inf_start = t.time()
        pred_map = run_inference(interpreter, input_tensor)
        inf_end = t.time()
        if inference_timer:
            inference_timer.end_inference()
        inf_lat = (inf_end-inf_start)*1000

        post_time = t.time()
        predicted_color = label_to_color_image(pred_map)
        ground_truth_color = get_ground_truth(mask_path, (w,h))
        post_lat = (t.time()-post_time)*1000

        logging.info(f"\n--- Image {image_id} ---")
        logging.info(f"Latencies (ms): Pre={pre_lat:.2f}, Inf={inf_lat:.2f}, Post={post_lat:.2f}")

        display_results(original_image, predicted_color, ground_truth_color)
        t.sleep(1)
        inference_timer.end_cycle() if inference_timer else None

    if inference_timer:
        inference_timer.flush()


# === MAIN ===
def main():
    parser = get_base_parser('Run Deeplab_v3 inference')
    args = parser.parse_args()
    run(mode=args.mode, model=args.model, size=args.size, delay=args.delay)

if __name__=="__main__":
    main()

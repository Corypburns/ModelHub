import os

import tensorflow as tf
import time as t
import cv2
import numpy as np
import logging

from CODEBASE.helper_functions import get_base_parser, load_model
from CODEBASE.config import *

IMAGE_CLASSIFICATION_FOLDER = MODEL_BASE_PATH / "Image-Classification"
TEST_IMAGE_PATH = DATA_SETS_PATH / "Image-Classification" / "test2017"
LABEL_MAP = LABEL_MAPS_PATH / "Image-Classification" / "labels.txt"

logger = logging.getLogger(__name__)

def load_labels(label_path):
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def run_inference(interpreter, labels, size, mode="CPU1", delay=0.5, inference_timer=None):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]

    image_paths = list(TEST_IMAGE_PATH.glob("*.jpg"))

    if size > 0:
        image_paths = image_paths[:size]

    for img_path in image_paths:
        cycle = inference_timer.start_cycle() if inference_timer else None
        t.sleep(delay)
        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            logger.warning("Could not read image %s. Skipping.", img_path)
            continue

        resized_img = cv2.resize(raw_img, (width, height))
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        input_dtype = input_details[0]["dtype"]
        if input_dtype in [np.uint8, tf.uint8]:
            input_tensor = np.expand_dims(rgb_img, axis=0).astype(np.uint8)
        elif input_dtype in [np.int8, tf.int8]:
            input_tensor = np.expand_dims(rgb_img.astype(np.float32) - 128.0, axis=0).astype(np.int8)
        else:
            input_tensor = np.expand_dims(rgb_img.astype(np.float32) / 255.0, axis=0)

        if inference_timer:
            inference_timer.start_inference()
        inf_start = t.time()

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        inf_end = t.time()

        if inference_timer:
            inference_timer.end_inference()

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_index = output.argmax()
        predicted_label = labels[predicted_index]
        confidence = output[predicted_index]

        inf_lat = (inf_end - inf_start) * 1000
        logger.info(
            "Image: %s | Pred: %s (%.2f%%) | Inf Lat: %.2f ms",
            img_path.name,
            predicted_label,
            confidence * 100,
            inf_lat,
        )

        t.sleep(1)
        if inference_timer:
            inference_timer.end_cycle()

def run(mode, model: str, size=0, delay=0.5, inference_timer=None):
    labels = load_labels(LABEL_MAP)
    model_path = IMAGE_CLASSIFICATION_FOLDER / f'{model}.tflite'
    if not os.path.isfile(model_path):
        logger.error("No .tflite model found %s", model_path)
        return

    logger.info(f"Model {model_path} Found")

    mode_map = {
        "CPU1": {"threads": 1, "str": "CPU1"},
        "CPU4": {"threads": 4, "str": "CPU4"},
        "GPU": {"threads": 0, "str": "GPU"},
    }

    if mode not in mode_map:
        logger.error("Invalid mode: %s. Choose from %s", mode, list(mode_map.keys()))
        return

    config = mode_map[mode]

    logger.info("%s", "=" * 20)
    logger.info("Running inference for model: %s", model_path.name)
    logger.info("%s", "=" * 20)

    interpreter = load_model(model_path, num_threads=config["threads"])
    run_inference(interpreter, labels, size, mode=config["str"], delay=delay, inference_timer=inference_timer)
    if inference_timer:
        inference_timer.flush()

def main():
    parser = get_base_parser('Run Image Classification Models')
    args = parser.parse_args()

    run(mode=args.mode, model=args.model, size=args.size, delay=args.delay)

if __name__ == "__main__":
    main()

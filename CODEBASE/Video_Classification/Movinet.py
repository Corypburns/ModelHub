import os

import tensorflow as tf
import time as t
import cv2, numpy as np
import logging
from CODEBASE.config import *
from CODEBASE.helper_functions import load_model, get_base_parser

logger = logging.getLogger(__name__)

# === CONFIG ===
TEST_VIDEO_PATH = DATA_SETS_PATH / "Video-Classification" / "kinetics_600_5000"
MODEL_DIR = MODEL_BASE_PATH / "Video-Classification"
LABELS_PATH = LABEL_MAPS_PATH / "Video-Classification" / "kinetics_600_labels.txt"

def load_labels(label_path):
    # Load model labels safely
    if label_path.exists():
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
    else:
        logger.warning("%s not found. Using generic labels.", label_path)
        labels = [f"Class_{i}" for i in range(600)]
    return labels

# === INFERENCE STEP ===
def video_classification_step(interpreter, labels, limit: int, delay=0.5, inference_timer=None):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = None
    height, width = 224, 224
    is_streaming = False

    for item in input_details:
        detail = item if isinstance(item, list) else item
        if not isinstance(detail, dict):
            continue

        shape = detail.get("shape", [])

        if len(shape) >= 4 and shape[-1] == 3:
            input_index = detail.get("index")
            if len(shape) == 5:
                logger.info("Detected streaming input shape %s", shape)
                height, width = int(shape[2]), int(shape[3])
                is_streaming = True
            elif len(shape) == 4:
                height, width = int(shape[1]), int(shape[2])
                is_streaming = False
            break

    if input_index is None:
        logger.critical("Could not find an input tensor with 3 color channels (RGB).")
        return

    output_index = None
    for item in output_details:
        detail = item if isinstance(item, list) else item
        if not isinstance(detail, dict):
            continue

        shape = detail.get("shape", [])
        if len(shape) > 0 and shape[-1] == len(labels):
            output_index = detail.get("index")
            break

    if output_index is None:
        first_out = output_details
        detail = first_out if isinstance(first_out, list) else first_out
        output_index = detail.get("index", 0)

    all_videos = list(TEST_VIDEO_PATH.glob("*.mp4"))
    if limit > 0:
        all_videos = all_videos[:limit * 10]

    logger.info("Starting inference on %d videos", len(all_videos))
    logger.info("-" * 50)

    for img_path in all_videos:
        inference_timer.start_cycle() if inference_timer else None
        t.sleep(delay)
        cap = cv2.VideoCapture(str(img_path))
        ret, raw_img = cap.read()
        cap.release()

        if not ret:
            logger.warning("Failed to read %s. Skipping.", img_path.name)
            continue

        resized_img = cv2.resize(raw_img, (width, height))
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        input_dtype = input_details[0]["dtype"]
        if input_dtype == np.uint8:
            input_tensor = np.expand_dims(rgb_img, axis=0).astype(np.uint8)
        else:
            input_tensor = np.expand_dims(rgb_img, axis=0).astype(np.float32)
            input_tensor = input_tensor / 255.0
        if is_streaming:
            input_tensor = np.expand_dims(input_tensor, axis=1)

        if inference_timer:
            inference_timer.start_inference()
        inf_start = t.time()

        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()

        inf_end = t.time()
        if inference_timer:
            inference_timer.end_inference()

        raw_output = interpreter.get_tensor(output_index)
        output = np.array(raw_output).flatten()
        predicted_index = output.argmax()

        if predicted_index < len(labels):
            predicted_label = labels[predicted_index]
        else:
            predicted_label = f"Unknown_{predicted_index}"

        confidence = output[predicted_index]
        inf_lat = (inf_end - inf_start) * 1000

        logger.info(
            "Video: %-25s | Pred: %-20s | Conf: %5.1f%% | Latency: %6.2f ms",
            img_path.name[:25],
            predicted_label[:20],
            confidence * 100,
            inf_lat,
        )
        inference_timer.end_cycle() if inference_timer else None

def run(mode, model, size=0, delay=0.5, inference_timer=None):
    labels = load_labels(LABELS_PATH)

    model_path = MODEL_DIR / f"{model}.tflite"

    if not os.path.isfile(model_path):
        logger.error("Model not found at: %s", model_path)
        return

    mode_thread_map = {"CPU1": 1, "CPU4": 4, "GPU": 0}

    if mode not in mode_thread_map:
        logger.error("Invalid mode: %s. Choices are %s", mode, list(mode_thread_map.keys()))
        return

    num_threads = mode_thread_map[mode]

    logger.info("%s", "=" * 70)
    logger.info("EVALUATING MODEL: %s", model)
    logger.info("%s", "=" * 70)

    try:
        interpreter = load_model(model_path, num_threads=num_threads)
        video_classification_step(interpreter, labels, limit=size, delay=delay, inference_timer=inference_timer)
        if inference_timer:
            inference_timer.flush()
        logger.info("Finished evaluation for %s", model_path.name)
    except Exception as e:
        logger.exception("Failed to evaluate %s", model)


def main():
    parser = get_base_parser('Run Video Classification Inference')
    args = parser.parse_args()
    run(mode=args.mode, model=args.model, size=args.size, delay=args.delay)

if __name__ == "__main__":
    main()
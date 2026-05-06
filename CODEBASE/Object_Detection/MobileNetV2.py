import os
import tensorflow as tf
import time as t
from pathlib import Path
import cv2
import numpy as np
import argparse
import logging

from model_hub.CODEBASE.config import *
from model_hub.CODEBASE.helper_functions import load_model, get_base_parser

# === CONFIG ===
TEST_IMAGE_PATH = DATA_SETS_PATH / 'Image-Classification'/ "test2017"
MODEL_FOLDER = BASE_PATH / "MODELBASE" / "Object-Detection"
LABEL_MAP = BASE_PATH / "LABELMAPS" / "Object-Detection" / "labelmap.txt"
delay = 0.5

# =========================================================
# LABELS
# =========================================================
def load_labels(label_path):
    if not Path(label_path).exists():
        logging.warning(f"{label_path} not found. Using generic labels.")
        return [f"Class_{i}" for i in range(100)]

    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]



# =========================================================
# DRAW DETECTIONS
# =========================================================
def draw_boxes(image, boxes, classes, scores, num_detections, labels):

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
        label = labels[class_id] if class_id < len(labels) else f"ID {class_id}"

        text = f"{label}: {scores[i]:.2f}"

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
        cv2.putText(image, text,
                    (x_min, y_min-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,0,0),
                    2)

    return image


# =========================================================
# INFERENCE
# =========================================================
def image_processing_inference(interpreter, labels, size, mode, visualize):

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    input_index = input_details["index"]
    input_shape = input_details["shape"]
    input_dtype = input_details["dtype"]

    input_h, input_w = input_shape[1:3]

    image_paths = sorted(TEST_IMAGE_PATH.glob("*.jpg"))

    if size > 0:
        image_paths = image_paths[:size*20]

    logging.info(f"\nRunning inference on {len(image_paths)} images")
    logging.info("-"*60)

    for img_path in image_paths:
        t.sleep(delay)

        img_raw = cv2.imread(str(img_path))
        if img_raw is None:
            continue

        img_resized = cv2.resize(img_raw, (input_w, input_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # input_tensor = np.expand_dims(img_rgb, axis=0).astype(np.float32)

        if input_dtype == np.uint8:
            # For UINT8 models: Do NOT normalize. Keep 0-255 range.
            input_tensor = np.expand_dims(img_rgb, axis=0).astype(np.uint8)
        else:
            input_tensor = np.expand_dims(img_rgb, axis=0).astype(np.float32)
            input_tensor = input_tensor / 255.0

        # ------------------ INFERENCE ------------------
        start = t.time()

        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()

        latency = (t.time() - start) * 1000

        # ------------------ OUTPUTS ------------------
        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]
        num_detections = interpreter.get_tensor(output_details[3]["index"])[0]

        logging.info(f"{img_path.name[:20]:<20} | Latency: {latency:6.2f} ms")

        drawn = draw_boxes(
            img_raw.copy(),
            boxes,
            classes,
            scores,
            num_detections,
            labels
        )

        if visualize and os.environ.get("DISPLAY"):
            cv2.imshow("Detections", drawn)
            cv2.waitKey(1)


def run(mode, model, size=0, visualize=False):
    """
    Programmatic entry point for TFLite model evaluation.
    
    Args:
        mode (str): "CPU1", "CPU4", or "GPU"
        size (int): Number of images (0 for all)
        visualize (bool): Whether to show the detection window
    """
    labels = load_labels(LABEL_MAP)

    mode_map = {
        "CPU1": {"threads": 1, "str": "CPU1"},
        "CPU4": {"threads": 4, "str": "CPU4"},
        "GPU":  {"threads": 0, "str": "GPU"},
    }

    if mode not in mode_map:
        logging.error(f"Invalid mode: {mode}. Choices are {list(mode_map.keys())}")
        return

    config = mode_map[mode]


    model_path = MODEL_FOLDER / f"{model}.tflite"
    if not os.path.isfile(model_path):
        logging.error("No .tflite model found at %s", model_path)
        return



    logging.info("\n" + "="*70)
    logging.info(f"EVALUATING: {model}")
    logging.info("="*70)

    try:
        interpreter = load_model(
            model_path,
            num_threads=config["threads"]
        )

        image_processing_inference(
            interpreter,
            labels,
            size,
            config["str"], 
            visualize
        )
        logging.info("\nFinished model.")
    except Exception as e:
        logging.error(f"Failed to evaluate {model}: {e}")

    # Clean up OpenCV windows if any were opened
    cv2.destroyAllWindows()

# === UPDATED MAIN FOR CLI ===
def main():
    parser = get_base_parser('Run Object Detection Models')
    args = parser.parse_args()

    # Call the run method with parsed arguments
    run(mode=args.mode, model=args.model, size=args.size, visualize=args.visualize)

if __name__ == "__main__":
    main()
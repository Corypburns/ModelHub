import logging
import os
import tensorflow as tf
import matplotlib.pyplot as mpl
import cv2
import time as t

from CODEBASE.config import *
from CODEBASE.helper_functions import load_model, get_base_parser

logger = logging.getLogger("esrgan_tf")

IMAGE_PATH = DATA_SETS_PATH / "Super-Resolution"
SUPER_RESOLUTION_FOLDER = MODEL_BASE_PATH / "Super-Resolution"
# === MODEL FINDER ===
def find_models(folder_path, extensions=("*.tflite",)):
    """Finds all model files in the specified folder based on extensions."""
    models = []
    if folder_path.exists():
        for ext in extensions:
            models.extend(folder_path.rglob(ext))
    return sorted(models)


# === PROCESSING & INFERENCE ===
def run_inference(interpreter, img_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    lr = cv2.imread(str(img_path))
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    lr = tf.expand_dims(lr, axis=0)
    lr = tf.cast(lr, tf.float32)

    interpreter.set_tensor(input_details[0]["index"], lr)

    start_inf = t.time()
    interpreter.invoke()
    end_inf = t.time()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    sr = tf.squeeze(output_data, axis=0)
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.round(sr)
    sr = tf.cast(sr, tf.uint8)

    lr_disp = tf.cast(tf.squeeze(lr, axis=0), tf.uint8)

    mpl.figure(figsize=(10, 4))
    mpl.subplot(1, 2, 1)
    mpl.title("ESRGAN (x4)")
    mpl.imshow(sr.numpy())

    bicubic = tf.image.resize(lr_disp, [200, 200], tf.image.ResizeMethod.BICUBIC)
    bicubic = tf.cast(bicubic, tf.uint8)

    mpl.subplot(1, 2, 2)
    mpl.title("Bicubic")
    mpl.imshow(bicubic.numpy())

    logger.info(
        "Inference on %s completed in %.2f ms",
        img_path.name,
        (end_inf - start_inf) * 1000,
    )

    mpl.show()

def run(mode="CPU1", model=None, size=None, delay=0.5, inference_timer=None):
    if mode == "CPU1":
        num_threads = 1
    elif mode == "CPU4":
        num_threads = 4
    else:
        num_threads = 0

    model_path = SUPER_RESOLUTION_FOLDER / f"{model}.tflite"


    if not os.path.isfile(model_path):
        logger.error("Model not found at:%s.", model_path)
        return

    image_files = sorted(list(IMAGE_PATH.glob("*.jpg")))
    if not image_files:
        logger.warning("No .jpg files found in %s", IMAGE_PATH)
        return

    if size is not None:
        image_files = image_files[:size * 40]

    logger.info("Mode: %s | Found %d image(s).", mode, len(image_files))

    logger.info("%s", "=" * 40)
    logger.info("Testing Model: %s", model)
    logger.info("%s", "=" * 40)

    try:
        interpreter = load_model(model_path, num_threads)
        for img_path in image_files:
            inference_timer.start_cycle() if inference_timer else None
            t.sleep(delay)
            run_inference(interpreter, img_path)
            inference_timer.end_cycle() if inference_timer else None
        if inference_timer:
            inference_timer.flush()
    except Exception as e:
        logger.error("Failed during inference for model %s: %s", model, e)

def main():
    parser = get_base_parser('Run Super-Resolution Inference')
    args = parser.parse_args()

    run(mode=args.mode, model=args.model, size=args.size, delay=args.delay)

if __name__ == "__main__":
    main()

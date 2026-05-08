import os
import tensorflow as tf
import time as t
import numpy as np
import pandas as pd
import logging
from CODEBASE.config import *
from CODEBASE.helper_functions import load_model, get_base_parser

DATASET_PATH = DATA_SETS_PATH/ "Text-Classification" / "WordVec" / "IMDB_Dataset.csv"
TEXT_CLASSIFICATION_FOLDER = MODEL_BASE_PATH / "Text-Classification"

delay = 0.5
logger = logging.getLogger(__name__)

# === MODEL FINDER ===
def find_models(folder_path, extensions=("*.tflite",)):
    """Finds all model files in the specified folder based on extensions."""
    models = []
    if folder_path.exists():
        for ext in extensions:
            models.extend(folder_path.rglob(ext))
    return sorted(models)

# === DATA + PREPROCESSING ===
def read_dataset():
    if not DATASET_PATH.exists():
        logger.error("Dataset not found at %s", DATASET_PATH)
        return None
    return pd.read_csv(DATASET_PATH)

def text_tokenizer(data):
    vocab = {word: i for i, word in enumerate(sorted(set(" ".join(data["review"]).split())))}
    return vocab

def truncation(seq, max_length: int, pad_value: int = 0, direction: str = "post"):
    if len(seq) > max_length:
        return seq[:max_length] if direction == "post" else seq[-max_length:]
    else:
        if direction == "post":
            return seq + [pad_value] * (max_length - len(seq))
        else:
            return [pad_value] * (max_length - len(seq)) + seq


# === INFERENCE ===
def text_classification_step(interpreter, data, vocab, size=None):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    max_vocab_length = 9999
    max_length = input_details[0]["shape"][1]

    reviews_to_process = data["review"]
    if size is not None:
        reviews_to_process = reviews_to_process[:size * 20]

    for idx, review in enumerate(reviews_to_process, start=1):
        t.sleep(delay)
        tokens = str(review).lower().split()
        id_seq = [min(vocab.get(w, 0), max_vocab_length) for w in tokens]
        id_seq = truncation(id_seq, max_length, direction="pre")
        input_array = np.array([id_seq], dtype=np.int32)

        inf_start = t.time()
        interpreter.set_tensor(input_details[0]["index"], input_array)
        interpreter.invoke()
        inf_end = t.time()

        outputs = interpreter.get_tensor(output_details[0]["index"])
        predicted_class = np.argmax(outputs)
        confidence = outputs[0][predicted_class] if outputs.ndim == 2 else outputs[predicted_class]
        inf_lat = (inf_end - inf_start) * 1000

        logger.info(
            "Review #%d | Prediction: Class %d (%.2f%%) | Inference Latency: %.2f ms",
            idx,
            predicted_class,
            confidence * 100,
            inf_lat,
        )

        snippet = str(review)[:80] + "..." if len(str(review)) > 80 else str(review)
        logger.info('  -> "%s"', snippet)

def run(mode="CPU1", model= None, size=None):
    if mode == "CPU1":
        num_threads = 1
    elif mode == "CPU4":
        num_threads = 4
    else:
        num_threads = 0

    logger.info("Loading dataset and building vocabulary...")
    data = read_dataset()
    if data is None:
        return
    vocab = text_tokenizer(data)

    model_path = TEXT_CLASSIFICATION_FOLDER / f"{model}.tflite"

    if not os.path.isfile(model_path):
        logger.error("Model not found at: %s", model_path)
        return

    logger.info("========================================")
    logger.info("Testing Model: %s", model)
    logger.info("========================================")

    try:
        interpreter = load_model(model_path, num_threads)
        text_classification_step(interpreter, data, vocab, size=size)
    except Exception as e:
        logger.error("Failed to process model %s: %s", model, e)
   

# === UPDATED MAIN FOR CLI ===
def main():
    parser = get_base_parser("Run Text Classification Inference")
    args = parser.parse_args()
    run(mode=args.mode, model=args.model, size=args.size)

if __name__ == "__main__":
    main()

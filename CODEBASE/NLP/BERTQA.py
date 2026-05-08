import os
import time as t
import numpy as np
import json
import logging
from transformers import BertTokenizer
from CODEBASE.helper_functions import load_model, get_base_parser
from CODEBASE.config import *

logger = logging.getLogger(__name__)

NLP_FOLDER = MODEL_BASE_PATH/ "NLP"
DATASET_PATH = DATA_SETS_PATH / "NLP" / "Train" / "train-v2.0.json"
delay = 0.5
# === MODEL FINDER ===
def find_models(folder_path, extensions=("*.tflite",)):
    """Finds all model files in the specified folder based on extensions."""
    models = []
    if folder_path.exists():
        for ext in extensions:
            models.extend(folder_path.rglob(ext))
    return sorted(models)

# === LOAD DATA & MODEL ===
def load_dataset():
    if not DATASET_PATH.exists():
        logger.error("Dataset not found at: %s", DATASET_PATH)
        return []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = [(qa["question"], paragraph["context"], qa["answers"][0]["text"] if qa["answers"] else "")
               for article in data["data"]
               for paragraph in article["paragraphs"]
               for qa in paragraph["qas"]]

    return samples


tokenizer = BertTokenizer.from_pretrained("google/mobilebert-uncased")

def encode(question, context, max_len=384):
    tokens = tokenizer(
        question,
        context,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    input_ids = tokens["input_ids"].astype(np.int32)
    attention_mask = tokens["attention_mask"].astype(np.int32)
    token_type_ids = tokens.get(
        "token_type_ids",
        np.zeros_like(input_ids, dtype=np.int32),
    ).astype(np.int32)

    return input_ids, attention_mask, token_type_ids

def predict(interpreter, inputs):
    input_ids, attention_mask, token_type_ids = inputs
    input_details = interpreter.get_input_details()

    for inp in input_details:
        name = inp["name"]

        if "input_ids" in name:
            interpreter.set_tensor(inp["index"], input_ids)
        elif "attention_mask" in name:
            interpreter.set_tensor(inp["index"], attention_mask)
        elif "segment" in name or "token_type" in name:
            interpreter.set_tensor(inp["index"], token_type_ids)

    interpreter.invoke()

    output_details = interpreter.get_output_details()
    start_logits = interpreter.get_tensor(output_details[0]["index"])
    end_logits = interpreter.get_tensor(output_details[1]["index"])

    return start_logits, end_logits

def get_answer(start_logits, end_logits, input_ids):
    start = np.argmax(start_logits)
    end = np.argmax(end_logits)
    if end < start or (end - start + 1) > 30:
        return ""
    tokens = input_ids[0][start:end + 1]
    return tokenizer.decode(tokens, skip_special_tokens=True)

# === INFERENCE ===
def bertqa_step(interpreter, samples, size=None):
    samples_to_process = samples
    if size is not None:
        samples_to_process = samples[:size]

    for idx, (question, context, true_answer) in enumerate(samples_to_process, start=1):
        t.sleep(delay)
        inputs = encode(question, context)
        inf_start = t.time()
        start_logits, end_logits = predict(interpreter, inputs)
        inf_end = t.time()

        pred_answer = get_answer(start_logits, end_logits, inputs[0])
        inf_lat = (inf_end - inf_start) * 1000

        logger.info("--- Inference Stats | Sample #%d ---", idx)
        logger.info("Q: %s", question)
        logger.info("Prediction: %s", pred_answer)
        logger.info("True Answer: %s", true_answer)
        logger.info("Inference Latency: %.2f ms", inf_lat)

def run(mode="CPU1", model=None, size=None):
    if mode == "CPU1":
        num_threads = 1
    elif mode == "CPU4":
        num_threads = 4
    elif mode == "GPU":
        num_threads = 0
    else:
        logger.error("Unsupported mode: %s", mode)
        return

    logger.info("Loading dataset...")
    samples = load_dataset()
    if not samples:
        logger.error("Dataset is empty or could not be loaded.")
        return

    model_path = NLP_FOLDER / f"{model}.tflite"
    if not os.path.isfile(model_path):
        logger.error("No .tflite model found at %s", model_path)
        return

    logger.info("Mode: %s | Found", mode)


    logger.info("%s", "=" * 40)
    logger.info("Testing Model: %s", model)
    logger.info("%s", "=" * 40)

    try:
        interpreter = load_model(model_path, num_threads)
        bertqa_step(interpreter, samples, size=size)
    except Exception as e:
        logger.error("Failed to run model %s: %s", model, e)

def main():
    parser = get_base_parser(description="Run NLP Inference")
    args = parser.parse_args()
    run(mode=args.mode, model=args.model, size=args.size)

if __name__ == "__main__":
    main()

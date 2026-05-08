import argparse
import os
from pathlib import Path
import tensorflow as tf
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    GPT2Tokenizer, GPT2LMHeadModel
)
import time as t
from datasets import load_dataset
import logging
import json
from CODEBASE.config import *
from CODEBASE.helper_functions import get_base_parser
def run(mode="CPU1", size=None, model="palm", delay=0.5, inference_timer=None):
    num_threads = 1 if mode == "CPU1" else 4 if mode == "CPU4" else 0

    if mode == "GPU":
        physical_gpus = tf.config.list_physical_devices('GPU')
        if physical_gpus:
            try:
                for gpu in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info("Using GPU with memory growth enabled.")
            except RuntimeError as e:
                logging.error(f"GPU config error: {e}")

    prompts = load_prompts(size=size, delay=delay)

    model_path = AUTOCOMPLETE_MODEL_PATH / f'{model}.tflite'

    if not os.path.isfile(model_path):
        logging.error(f"Model not found at: {model_path}")
        return


    model_name = model
    logging.info(f"\n{'='*40}\nTesting Model: {model_name}\n{'='*40}")

    tokenizer, model_loaded = load_model(model_path, model, num_threads)
    run_inference(tokenizer, model_loaded, prompts, size, delay=delay, inference_timer=inference_timer)
    if inference_timer:
        inference_timer.flush()

def main():
    """Parses command-line arguments and starts the autocomplete process."""
    parser = get_base_parser(description="Unified Autocomplete")
    args = parser.parse_args()

    run(
        mode=args.mode,
        size=args.size,
        model=args.model,
        delay=args.delay,
    )

if __name__ == "__main__":
    main()

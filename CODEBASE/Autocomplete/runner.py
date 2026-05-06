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
from model_hub.CODEBASE.config import *
from model_hub.CODEBASE.helper_functions import get_base_parser
delay = 0.5

# === CONFIG ===
LOCAL_DATASET_FILE =  DATA_SETS_PATH / 'Autocomplete' / "prompts.json"
AUTOCOMPLETE_MODEL_PATH = MODEL_BASE_PATH / "Autocomplete"

# === MODEL FINDER ===
def find_models(folder_path):
    """Finds all Hugging Face model directories within a specified folder."""
    models = []
    if folder_path.exists():
        for p in folder_path.iterdir():
            if p.is_dir() and (p / "config.json").exists():
                models.append(p)
    return sorted(models)

# === LOAD PROMPTS ===
def load_prompts(size=None, max_size= 1000):
    """
    Loads prompts from a local JSON file or streams them from Hugging Face.

    Args:
        size (int, optional): The number of prompts to return. Defaults to None.
        max_size (int, optional): The maximum number of prompts to download and cache. Defaults to 1000.

    Returns:
        list: A list of prompt strings.
    """
    if LOCAL_DATASET_FILE.exists():
        with open(LOCAL_DATASET_FILE, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if size and len(prompts) >= max_size:
            logging.info(f"Loading prompts from local file: {LOCAL_DATASET_FILE}")
            prompts = prompts[:size]
            return prompts

    logging.info("Streaming dataset from Hugging Face...")
    dataset = load_dataset("amazon/AmazonQAC", split="train", streaming=True)

    prompts = []
    for i, item in enumerate(dataset):
        # print(item)
        t.sleep(delay)
        if 'final_search_term' in item:
            target_text = item['final_search_term']
            prompts.append( target_text)
        if len(prompts) >= max_size:
            break

    LOCAL_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    return prompts

# === LOAD MODEL ===
def load_model(model_path, model_type="seq2seq", num_threads=0):
    logging.info(f"Loading {model_type} model from: {model_path}")
    """
    Loads a Hugging Face model and tokenizer.

    Args:
        model_path (Path or str): The path or ID of the model.
        model_type (str, optional): The type of model ('gpt2' or 'seq2seq'). Defaults to "seq2seq".
        num_threads (int, optional): Number of threads for TensorFlow. Defaults to 0.

    Returns:
        tuple: The tokenizer and the model.
    """
    if num_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)

    model_id = str(model_path) if isinstance(model_path, Path) else model_path

    if model_type == "Keras_GPT2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side="left")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        # GPT2 needs a pad token defined
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base', padding_side="left")
        model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')

    return tokenizer, model

# === RUN INFERENCE ===
def run_inference(tokenizer, model, prompts, size: int):
    # We no longer use 'size' here to loop. We just loop through the prompts list.
    """
    Runs autocomplete inference for a list of prompts.

    Args:
        tokenizer: The model's tokenizer.
        model: The loaded language model.
        prompts (list): A list of input prompts.
        size (int): The number of prompts to process from the list.
    """
    for idx, prompt_data in enumerate(prompts):
        if idx == size:
            break
        if isinstance(prompt_data, (list, tuple)):
            prompt = prompt_data
        else:
            prompt = str(prompt_data)
        if not prompt:
            continue

        logging.info(f"=== Prompt #{idx} ===\nInput: {prompt}")

        # Fix: Capture attention_mask and use return_tensors="pt"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        start_time = t.time()
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id 
        )
        end_time = t.time()
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        logging.info(f"Output: {generated_text}")
        logging.info(f"Inference Latency: {(end_time - start_time)*1000:.2f} ms")

def run(mode="CPU1", size=None, model="palm"):
    """
    Sets up the environment and runs the autocomplete inference loop.

    Args:
        mode (str, optional): Execution mode ('CPU1', 'CPU4', 'GPU'). Defaults to "CPU1".
        size (int, optional): Number of prompts to process. Defaults to None (all).
        model_type (str, optional): Type of model to use ('gpt2', 'palm'). Defaults to "palm".
        model_folder (str, optional): Path to a folder containing local models. Defaults to None.
    """
    # 1. Thread settings / GPU config
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

    # 2. Load Prompts
    prompts = load_prompts(size=size)
    
    # 3. Resolve Model Paths
    model_path = AUTOCOMPLETE_MODEL_PATH / f'{model}.tflite'
    
    if not os.path.isfile(model_path):
        logging.error(f"Model not found at: {model_path}")
        return
    

    # 4. Loop through models and run inference
    model_name = model
    logging.info(f"\n{'='*40}\nTesting Model: {model_name}\n{'='*40}")
    
    tokenizer, model_loaded = load_model(model_path, model, num_threads)
    run_inference(tokenizer, model_loaded, prompts, size)

def main():
    """Parses command-line arguments and starts the autocomplete process."""
    parser = get_base_parser(description="Unified Autocomplete")
    args = parser.parse_args()

    run(
        mode=args.mode, 
        size=args.size, 
        model=args.model, 
    )

if __name__ == "__main__":
    main()

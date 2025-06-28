import tensorflow as tf
import numpy as np
import pandas as pd
import os
from transformers import BertTokenizer  # Optional: for decoding token IDs
from datetime import datetime as dt

# === Path setup ===
now = dt.now().strftime("%y-%m-%d_%H-%M-%S")
output = "/home/cory/code/Anik-Research-Testing/ModelHub/OUTPUTS/Autocomplete/OUTPUTS/Autocomplete-NLP-Outputs"
joined_output = os.path.join(output, f"Autocomplete-NLP_{now}")
dataset = "/home/cory/code/Anik-Research-Testing/DATASETS/Text-Based Datasets/Autocomplete/search_data_dataset.csv"
model = "/home/cory/code/Anik-Research-Testing/ModelHub/MODELS/AutoComplete-GenAI/autocomplete.tflite"
os.makedirs(joined_output, exist_ok=True)

# === Load dataset ===
df = pd.read_csv(dataset)

# === Load TFLite model ===
interpreter = tf.lite.Interpreter(model_path=model)
interpreter.allocate_tensors()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
from datetime import datetime as dt

# === Path setup ===
now = dt.now().strftime("%y-%m-%d_%H-%M-%S")
output_base = "/home/cory/code/Anik-Research-Testing/ModelHub/OUTPUTS/Autocomplete/OUTPUTS/Autocomplete-NLP-Outputs"
joined_output = os.path.join(output_base, f"Autocomplete-NLP_{now}")
dataset = "/home/cory/code/Anik-Research-Testing/DATASETS/Text-Based Datasets/Autocomplete/search_data_dataset.csv"
os.makedirs(joined_output, exist_ok=True)

# === Load dataset ===
df = pd.read_csv(dataset)

# === Load pre-trained autocomplete model ===
model_name = "distilgpt2"  # or try "gpt2" or "EleutherAI/gpt-neo-125M" if you want better results
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# === Run inference on top 10 samples ===
results = []
for i, row in df.head(10).iterrows():
    input_text = row['search_term']

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    full_prediction = generated
    predicted_continuation = generated[len(input_text):].strip()
    results.append((input_text, predicted_continuation, full_prediction))

# === Save results ===
results_df = pd.DataFrame(results, columns=["Input", "PredictedContinuation", "AutocompleteSuggestion"])
output_file = os.path.join(joined_output, f"autocomplete_hf_results_{now}.csv")
results_df.to_csv(output_file, index=False)

print(f"\n✅ Saved predictions to: {output_file}")

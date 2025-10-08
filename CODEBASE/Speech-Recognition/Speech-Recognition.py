# speech_inference_gpu_cpu.py
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import vosk
import soundfile as sf
import pandas as pd
from datetime import datetime as dt
import csv
import os

# === CONFIG ===
BASE_PATH = Path(r"E:\Code\Python\DATASETS\AUDIO")
LOG_DIR = Path(r"E:\Code\Python\ModelHub\OUTPUTS\Speech-Recognition")
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
OUTPUT_PATH = LOG_DIR / f"log_SER_{DATE_TIME}.csv"
HEADERS = ["Timestamp", "AudioFile", "Sound_Class", "Transcription"]

TARGET_SR = 16000

# === FUNCTIONS ===
def preprocess_audio(file_path, target_sr=TARGET_SR):
    y, sr = librosa.load(file_path, sr=target_sr)
    return y

def load_yamnet():
    print("Loading YAMNet model from TF Hub...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("YAMNet loaded!")
    return model

def load_class_map():
    class_map_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv',
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )
    return pd.read_csv(class_map_path)

def load_vosk_model(model_path=r"E:\Code\Python\ModelHub\CODEBASE\Speech-Recognition/models"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vosk model not found at {model_path}")
    return vosk.Model(model_path)

def run_inference(device="CPU1"):
    # === TensorFlow device setup ===
    physical_devices = tf.config.list_physical_devices('CPU')
    if device.startswith("CPU"):
        num_threads = 1 if device=="CPU1" else 4
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        print(f"Running inference on {device} with {num_threads} threads...")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("Running inference on GPU...")
        else:
            print("GPU not found, running on CPU1")
            tf.config.threading.set_intra_op_parallelism_threads(1)

    # === Load models ===
    yamnet_model = load_yamnet()
    class_map = load_class_map()
    vosk_model = load_vosk_model()

    # === CSV setup ===
    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)

    # === Process files ===
    for audio_file in BASE_PATH.rglob("*.wav"):
        print(f"\nProcessing: {audio_file.name}")

        # --- YAMNet inference ---
        waveform = preprocess_audio(audio_file).astype(np.float32)
        scores, embeddings, spectrogram = yamnet_model(waveform)
        predicted_index = np.argmax(scores)
        sound_class = class_map.iloc[predicted_index]['display_name']

        # --- Vosk transcription ---
        wf = sf.SoundFile(audio_file)
        rec = vosk.KaldiRecognizer(vosk_model, wf.samplerate)
        for block in wf.blocks(blocksize=4000, dtype='int16'):
            rec.AcceptWaveform(block)
        transcription_result = rec.FinalResult()
        transcription_text = vosk.json.loads(transcription_result).get("text", "")

        print(f"Sound Class: {sound_class}")
        print(f"Transcription: {transcription_text}")

        # --- Append to CSV ---
        with open(OUTPUT_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                             audio_file.name, sound_class, transcription_text])

# === MENU ===
def menu():
    print("\n--- Select Mode ---")
    print("1: CPU 1 Thread")
    print("2: CPU 4 Threads")
    print("3: GPU")
    print("4: Quit\n")
    choice = input("Enter your choice: ").strip()
    if choice=="1":
        run_inference("CPU1")
    elif choice=="2":
        run_inference("CPU4")
    elif choice=="3":
        run_inference("GPU")
    elif choice=="4":
        exit()
    else:
        print("Invalid choice")
        menu()

# === MAIN ===
if __name__=="__main__":
    print(f"Host: {BASE_PATH}")
    menu()

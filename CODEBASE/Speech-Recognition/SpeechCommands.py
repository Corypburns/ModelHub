from datetime import datetime as dt
from pathlib import Path
import torchaudio
import torch
import numpy as np
import time as t
import socket

# === HOST CONFIG ===
host = socket.gethostname()
print(f"Host: {host}")

match host:
    case "DESKTOP-FI8GT7F":
        BASE_PATH = Path(r"E:\Code\Python\ModelHub")
        TEST_AUDIO_BASE_PATH = Path(r"E:\Code\Python\DATASETS\GoogleSpeechCommands")
    case "nano1-desktop":
        BASE_PATH = Path("/home/nano1/anik-lab/ModelHub")
        TEST_AUDIO_BASE_PATH = Path("/home/nano1/anik-lab/coco")

LABEL_MAP = BASE_PATH / "LABELMAPS" / "Speech-Recognition" / "conv_actions_labels.txt"
LOG_DIR = BASE_PATH / "OUTPUTS" / "Speech-Recognition"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
OUTPUT_PATH = LOG_DIR / f"log_GSC_{DATE_TIME}.csv"

HEADERS = (
    "Timestamp,Review,Mode,"
    "Pre_Lat_ms,Inf_Lat_ms,Post_Lat_ms,"
    "Pre_E_mJ,Inf_E_mJ,Post_E_mJ,"
    "Pre_Max_V,Pre_Mean_V,Pre_Max_C,Pre_Mean_C,"
    "Inf_Max_V,Inf_Mean_V,Inf_Max_C,Inf_Mean_C,"
    "Post_Max_V,Post_Mean_V,Post_Max_C,Post_Mean_C,"
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW,Memory_mB\n"
)

# === CSV UTILS ===
def init_csv():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'w') as f:
            f.write(HEADERS)

def append_csv_row(*args):
    with open(OUTPUT_PATH, 'a') as f:
        f.write(",".join(map(str, args)) + "\n")

# === LABELS ===
def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# === DUMMY STATS ===
def dummy_stats():
    return (0, 0, 0, 0, 0)

# === LOAD MODEL (PyTorch) ===
def load_pytorch_model(mode):
    print(f"\n=== PyTorch Mode: {mode} ===")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model()
    model.eval()

    device = torch.device("cuda" if mode == "GPU" and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sample_rate = bundle.sample_rate

    print(f"[INFO] Loaded Wav2Vec2 Base ASR model on {device} ({sample_rate} Hz)")
    return model, bundle, device


# === PREPROCESS AUDIO ===
def preprocess_audio(wav_path, sample_rate, device):
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(device)
    return waveform

# === RUN INFERENCE (PyTorch CPU/GPU) ===
def run_pytorch_inference(labels, mode="CPU1"):
    print(f"\n=== PyTorch Mode: {mode} ===")
    model, bundle, device = load_pytorch_model(mode)
    sample_rate = bundle.sample_rate
    alphabet = bundle.get_labels()

    for wav_path in TEST_AUDIO_BASE_PATH.rglob("*.wav"):
        pre_max_v, pre_mean_v, pre_max_c, pre_mean_c, pre_pwr = dummy_stats()
        pre_time = t.time()
        waveform = preprocess_audio(wav_path, sample_rate, device)
        inf_start = t.time()

        with torch.inference_mode():
            emissions, _ = model(waveform)

        inf_end = t.time()
        post_time = t.time()

        tokens = torch.argmax(emissions, dim=-1)
        transcript = "".join([alphabet[i] for i in tokens[0]]).replace("|", " ").strip()

        pre_lat = (inf_start - pre_time) * 1000
        inf_lat = (inf_end - inf_start) * 1000
        post_lat = (post_time - inf_end) * 1000

        print(f"{wav_path.name} | Transcript: {transcript} | {inf_lat:.2f} ms")

        append_csv_row(
            dt.now().strftime("%Y-%m-%d %H:%M:%S"), wav_path.name, mode,
            f"{pre_lat:.4f}", f"{inf_lat:.4f}", f"{post_lat:.4f}",
            0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0
        )

# === MENU ===
def menu():
    print("\n--- Select Mode ---\n1: CPU 1 Thread\n2: CPU 4 Threads\n3: GPU (PyTorch)\n4: Quit\n")
    choice = input("Enter choice: ").strip()
    if choice == '1':
        run_pytorch_inference(labels, "CPU1")
    elif choice == '2':
        run_pytorch_inference(labels, "CPU4")
    elif choice == '3':
        run_pytorch_inference(labels, "GPU")
    elif choice == '4':
        exit()
    else:
        print("Invalid choice.")
        menu()

# === MAIN ===
def main():
    global labels
    init_csv()
    labels = load_labels(LABEL_MAP)
    menu()

if __name__ == "__main__":
    main()

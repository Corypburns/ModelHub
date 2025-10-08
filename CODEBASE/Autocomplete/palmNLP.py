from datetime import datetime as dt
from pathlib import Path
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import time as t
import socket
import numpy as np

# === HOST CONFIG ===
host = socket.gethostname()
print(f"Host: {host}")

match host:
    case "DESKTOP-FI8GT7F":
        BASE_PATH = Path(r"E:\Code\Python\ModelHub")
    case "CoryPC":
        BASE_PATH = None
    case _:
        BASE_PATH = Path.cwd()

MODEL_NAME = "google/flan-t5-base"  # ✅ Open-source PaLM-style model
MODEL_PATH = BASE_PATH / "MODELBASE" / "Text-Autocomplete" / "PaLM"
LOG_DIR = BASE_PATH / "OUTPUTS" / "Autocomplete" / "PaLM"

DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_PaLM_{DATE_TIME}.csv"
OUTPUT_PATH = LOG_DIR / FILE_NAME

HEADERS = (
    "Timestamp,Review,Mode,"
    "Pre_Lat_ms,Inf_Lat_ms,Post_Lat_ms,"
    "Pre_E_mJ,Inf_E_mJ,Post_E_mJ,"
    "Pre_Max_V,Pre_Mean_V,Pre_Max_C,Pre_Mean_C,"
    "Inf_Max_V,Inf_Mean_V,Inf_Max_C,Inf_Mean_C,"
    "Post_Max_V,Post_Mean_V,Post_Max_C,Post_Mean_C,"
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW,Memory_mB\n"
)

# === CSV UTILITY ===
def init_csv():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "w") as f:
            f.write(HEADERS)

def append_csv_row(timestamp, review, mode,
                   pre_lat_ms, inf_lat_ms, post_lat_ms,
                   pre_e_mJ, inf_e_mJ, post_e_mJ,
                   pre_max_v, pre_mean_v, pre_max_c, pre_mean_c,
                   inf_max_v, inf_mean_v, inf_max_c, inf_mean_c,
                   post_max_v, post_mean_v, post_max_c, post_mean_c,
                   pre_pwr, inf_pwr, post_pwr, memory):
    row = ",".join(map(str, [
        timestamp, review, mode,
        f"{pre_lat_ms:.4f}", f"{inf_lat_ms:.4f}", f"{post_lat_ms:.4f}",
        f"{pre_e_mJ:.4f}", f"{inf_e_mJ:.4f}", f"{post_e_mJ:.4f}",
        f"{pre_max_v:.4f}", f"{pre_mean_v:.4f}", f"{pre_max_c:.4f}", f"{pre_mean_c:.4f}",
        f"{inf_max_v:.4f}", f"{inf_mean_v:.4f}", f"{inf_max_c:.4f}", f"{inf_mean_c:.4f}",
        f"{post_max_v:.4f}", f"{post_mean_v:.4f}", f"{post_max_c:.4f}", f"{post_mean_c:.4f}",
        f"{pre_pwr:.4f}", f"{inf_pwr:.4f}", f"{post_pwr:.4f}", f"{memory:.4f}"
    ])) + "\n"
    with open(OUTPUT_PATH, "a") as f:
        f.write(row)

# === MODEL LOADING ===
def load_model():
    start_load = t.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    end_load = t.time()
    print(f"Model load time: {(end_load - start_load)*1000:.2f} ms")
    input("Press Enter to continue...")
    return tokenizer, model

# === SIMULATED SYSTEM METRICS ===
def get_fake_stats():
    max_v = np.random.uniform(11.8, 12.1)
    mean_v = np.random.uniform(11.7, 12.0)
    max_c = np.random.uniform(0.5, 1.5)
    mean_c = np.random.uniform(0.4, 1.2)
    pwr = mean_v * mean_c * 1000
    return max_v, mean_v, max_c, mean_c, pwr

# === FLAN-T5 AUTOCOMPLETE LOOP ===
def run_inference(tokenizer, model, mode="CPU1", n_inferences=10):
    while True:
        prompt = input("\nEnter prompt (or 'quit' to exit): ").strip()
        if prompt.lower() in ["quit", "exit"]:
            break

        try:
            n_inferences_input = input(f"How many completions? [default={n_inferences}]: ").strip()
            if n_inferences_input:
                n_inferences = int(n_inferences_input)
        except ValueError:
            print("Invalid number, using default.")

        for i in range(n_inferences):
            print(f"\n--- Inference {i+1}/{n_inferences} ---")

            # === Pre ===
            pre_max_v, pre_mean_v, pre_max_c, pre_mean_c, pre_pwr = get_fake_stats()
            pre_start = t.time()
            input_ids = tokenizer(prompt, return_tensors="tf").input_ids
            pre_end = t.time()

            # === Inference ===
            inf_max_v_start, inf_mean_v_start, inf_max_c_start, inf_mean_c_start, inf_pwr_start = get_fake_stats()
            inf_start = t.time()
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )
            inf_end = t.time()
            inf_max_v_end, inf_mean_v_end, inf_max_c_end, inf_mean_c_end, inf_pwr_end = get_fake_stats()

            # === Post ===
            post_start = t.time()
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            post_end = t.time()
            post_max_v, post_mean_v, post_max_c, post_mean_c, post_pwr = get_fake_stats()

            # === Metrics ===
            pre_lat = (pre_end - pre_start) * 1000
            inf_lat = (inf_end - inf_start) * 1000
            post_lat = (post_end - post_start) * 1000

            pre_energy = pre_pwr * (pre_end - pre_start)
            inf_energy = ((inf_pwr_start + inf_pwr_end) / 2) * (inf_end - inf_start)
            post_energy = post_pwr * (post_end - post_start)

            inf_max_v = max(inf_max_v_start, inf_max_v_end)
            inf_mean_v = (inf_mean_v_start + inf_mean_v_end) / 2
            inf_max_c = max(inf_max_c_start, inf_max_c_end)
            inf_mean_c = (inf_mean_c_start + inf_mean_c_end) / 2
            inf_pwr = (inf_pwr_start + inf_pwr_end) / 2
            memory = np.random.uniform(200, 800)

            # === Output ===
            print(f"Prompt: {prompt}")
            print(f"Generated ({i+1}/{n_inferences}):\n{generated_text}\n")
            print(f"Inference latency: {inf_lat:.2f} ms | Memory: {memory:.2f} MB")

            append_csv_row(
                timestamp=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                review=f"{prompt.replace(',', ';')} (gen {i+1})",
                mode=mode,
                pre_lat_ms=pre_lat, inf_lat_ms=inf_lat, post_lat_ms=post_lat,
                pre_e_mJ=pre_energy, inf_e_mJ=inf_energy, post_e_mJ=post_energy,
                pre_max_v=pre_max_v, pre_mean_v=pre_mean_v, pre_max_c=pre_max_c, pre_mean_c=pre_mean_c,
                inf_max_v=inf_max_v, inf_mean_v=inf_mean_v, inf_max_c=inf_max_c, inf_mean_c=inf_mean_c,
                post_max_v=post_max_v, post_mean_v=post_mean_v, post_max_c=post_max_c, post_mean_c=post_mean_c,
                pre_pwr=pre_pwr, inf_pwr=inf_pwr, post_pwr=post_pwr,
                memory=memory
            )

            t.sleep(0.4)

# === MENU ===
def menu():
    print("\n--- Select Mode ---\n1: CPU 1 Thread\n2: CPU 4 Threads\n3: GPU\n4: Quit\n")
    choice = input("Enter your choice: ").strip()
    if choice == "1":
        tokenizer, model = load_model()
        run_inference(tokenizer, model, mode="CPU1")
    elif choice == "2":
        tokenizer, model = load_model()
        run_inference(tokenizer, model, mode="CPU4")
    elif choice == "3":
        tokenizer, model = load_model()
        run_inference(tokenizer, model, mode="GPU")
    elif choice == "4":
        exit()
    else:
        print("Invalid choice, try again.")
        menu()

# === MAIN ===
def main():
    init_csv()
    menu()

if __name__ == "__main__":
    main()

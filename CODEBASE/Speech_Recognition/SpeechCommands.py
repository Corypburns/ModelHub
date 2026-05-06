import os
import random
import time
import numpy as np
import tensorflow as tf
import soundfile as sf
import logging

from model_hub.CODEBASE.helper_functions import get_base_parser
from  model_hub.CODEBASE.config import *


logger = logging.getLogger(__name__)

# === PATHS ===
TEST_AUDIO_BASE_PATH = DATA_SETS_PATH/ "Speech-Recognition"
SPEECH_RECOGNITION_FOLDER = MODEL_BASE_PATH/ "Speech-Recognition"
LABEL_MAP = LABEL_MAPS_PATH / "Speech-Recognition" / "conv_actions_labels.txt"
delay = 0.5

# === MODEL FINDER ===
def find_models(folder_path):
    return sorted(folder_path.rglob("*.tflite"))


# === LABELS ===
def load_labels(label_path):
    if not label_path.exists():
        logger.warning("Label map not found at %s", label_path)
        return []
    with open(label_path, "r") as f:
        return [line.strip() for line in f]


# === PREPROCESS AUDIO ===
def preprocess_audio(wav_path, target_sample_rate, input_details):
    waveform, sr = sf.read(str(wav_path), dtype="int16")

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1).astype(np.int16)

    if sr != target_sample_rate:
        waveform = tf.signal.resample(
            tf.convert_to_tensor(waveform, dtype=tf.float32),
            int(len(waveform) * target_sample_rate / sr),
        ).numpy().astype(np.int16)

    desired_length = 16000
    if len(waveform) < desired_length:
        waveform = np.pad(waveform, (0, desired_length - len(waveform)))
    else:
        waveform = waveform[:desired_length]

    waveform = waveform.astype(np.float32)

    declared_shape = input_details[0]["shape"]
    if len(declared_shape) == 1:
        pass
    elif declared_shape[-1] == 1:
        waveform = waveform.reshape(declared_shape)
    else:
        waveform = waveform.reshape(declared_shape)

    return waveform


# === LOAD TFLITE MODEL ===
def load_model(model_path, mode="CPU1"):
    logger.info("Loading TFLite model: %s in mode %s", model_path, mode)

    if mode == "GPU":
        try:
            delegate = tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so")
            interpreter = tf.lite.Interpreter(
                model_path=str(model_path),
                experimental_delegates=[delegate],
            )
            logger.info("Using GPU delegate")
        except ValueError:
            logger.warning("GPU delegate not available, falling back to CPU")
            interpreter = tf.lite.Interpreter(
                model_path=str(model_path),
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            )
    else:
        num_threads = 4 if mode == "CPU4" else 1
        interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
            num_threads=num_threads,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        logger.info("Using CPU with %d thread(s)", num_threads)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


# === RUN INFERENCE ===
def run_tflite_inference(interpreter, input_details, output_details, labels, audio_files):
    audio_input = next(
        t for t in input_details if "sample_data" in t["name"] and t["dtype"] == np.float32
    )
    rate_input = next(
        t for t in input_details if "sample_data" in t["name"] and t["dtype"] == np.int32
    )

    logger.info("Audio input : idx=%d shape=%s", audio_input["index"], audio_input["shape"])
    logger.info("Rate input  : idx=%d shape=%s", rate_input["index"], rate_input["shape"])

    for wav_path in audio_files:
        time.sleep(delay)
        waveform = preprocess_audio(wav_path, target_sample_rate=16000, input_details=input_details)

        interpreter.set_tensor(audio_input["index"], waveform)
        interpreter.set_tensor(rate_input["index"], np.array([16000], dtype=np.int32))
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])
        pred_index = int(np.argmax(output_data))
        pred_label = labels[pred_index] if labels else str(pred_index)

        logger.info("%s | Predicted: %s", wav_path.name, pred_label)


def run(mode="CPU1", model=None, size=None):
    labels = load_labels(LABEL_MAP)

    model_path = SPEECH_RECOGNITION_FOLDER / f"{model}.tflite"
    if not os.path.isfile(model_path):
        logger.error("Model not found at: %s", model_path)
        return

    audio_files = list(TEST_AUDIO_BASE_PATH.rglob("*.wav"))
    if not audio_files:
        logger.error("No .wav files found in %s", TEST_AUDIO_BASE_PATH)
        return

    random.shuffle(audio_files)
    if size is not None:
        audio_files = audio_files[:size*20]

    logger.info("--- Testing Model: %s ---", model_path.name)
    try:
        interpreter, input_details, output_details = load_model(model_path, mode)
        run_tflite_inference(interpreter, input_details, output_details, labels, audio_files)
    except Exception as e:
        logger.error("Failed to run model %s: %s", model_path.name, e, exc_info=True)


def main():
    parser = get_base_parser("Run Speech Recognition Inference")
    args = parser.parse_args()
    run(mode=args.mode, model=args.model, size=args.size)


if __name__ == "__main__":
    main()

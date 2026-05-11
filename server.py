from datetime import datetime
import socket
import json
import time
import traceback
from CODEBASE.helper_functions import EnergyMonitor, InferenceTimer
from CODEBASE.Autocomplete.runner import run as autocomplete_run
from CODEBASE.Image_Classification.img_classification_runner import run as image_classification_run
from CODEBASE.Image_Segmentation.Deeplab_v3 import run as image_segmentation_run
from CODEBASE.Object_Detection.MobileNetV2 import run as object_detection_run
from CODEBASE.NLP.BERTQA import run as nlp_run
from CODEBASE.Speech_Recognition.SpeechCommands import run as speech_recognition_run
from CODEBASE.Super_Resolution.ESRGAN_TF import run as super_resolution_run
from CODEBASE.Text_Classification.WordVec import run as text_classification_run
from CODEBASE.Video_Classification.Movinet import run as video_classification_run
from Non_AI_runner import run as non_ai_runner
from enum import Enum
from jtop import jtop

HOST = "0.0.0.0"
PORT = 50007



class ModelType(Enum):
    AUTOCOMPLETE = autocomplete_run
    IMAGE_CLASSIFICATION = image_classification_run
    IMAGE_SEGMENTATION = image_segmentation_run
    OBJECT_DETECTION = object_detection_run
    NLP = nlp_run
    SPEECH_RECOGNITION = speech_recognition_run
    SUPER_RESOLUTION = super_resolution_run
    TEXT_CLASSIFICATION = text_classification_run
    VIDEO_CLASSIFICATION = video_classification_run
    Non_AI = non_ai_runner

id_to_app  = [
    {'name': 'Autocomplete', 'run': ModelType.AUTOCOMPLETE},
    {'name': 'Image Classification', 'run': ModelType.IMAGE_CLASSIFICATION},
    {'name': 'Image Segmentation', 'run': ModelType.IMAGE_SEGMENTATION},
    {'name': 'Object Detection', 'run': ModelType.OBJECT_DETECTION},
    {'name': 'NLP', 'run': ModelType.NLP},
    {'name': 'Speech Recognition', 'run': ModelType.SPEECH_RECOGNITION},
    {'name': 'Super Resolution', 'run': ModelType.SUPER_RESOLUTION},
    {'name': 'Text Classification', 'run': ModelType.TEXT_CLASSIFICATION},
    {'name': 'Video Classification', 'run': ModelType.VIDEO_CLASSIFICATION},
    {'name': 'Non_AI', 'run': ModelType.Non_AI}
]

def run_ai(app_id, size, mode, model, delay, ts):
    app = id_to_app[app_id]
    inference_timer = InferenceTimer(app["name"].replace(" ", "_"), model, mode, ts)
    with jtop(0.4) as jt:
        monitor = EnergyMonitor(jt, interval=0.5, output_file=f"logs/{app['name'].replace(' ', '_')}/{model}_{mode}.csv")
        monitor.start()
        try:
            print(f"Running app {app['name']} with size {size}")
            start = time.perf_counter()
            app['run'](size=size, model=model, mode=mode, delay=delay, inference_timer=inference_timer)
            duration = time.perf_counter() - start
        finally:
            monitor.stop()
            monitor.join()

    return duration


def handle_client(conn, ts):
    with conn:
        print("Client connected")
       
        while True:
            data = conn.recv(4096)
            if not data:
                break
            try:
                msg = json.loads(data.decode())
                if msg["cmd"] == "run":                    
                    duration = run_ai(msg["app_id"], msg["size"], msg["mode"], msg['model'], delay=msg.get("delay", 0), ts=ts)

                    conn.sendall(json.dumps({
                        "status": "finished",
                        "duration": duration
                    }).encode())
                elif msg["cmd"] == "exit":
                    print('Client disconnected. Exiting')
                    return
            except Exception as e:
                traceback.print_exc()
                print(e)


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    print("Jetson server listening...")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    while True:
        conn, addr = s.accept()
        handle_client(conn, ts)
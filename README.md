# ModelHub

Energy fingerprinting system for AI and Non-AI workloads on Jetson edge devices.

## Setup

### Required directories (not stored in git — populate before running)

```
DATASETS/
├── Autocomplete/prompts.json
├── Image-Classification/test2017/       # .jpg files
├── Image-Segmentation/VOC_2012/VOC2012_train_val/
│   ├── ImageSets/Segmentation/val.txt
│   ├── JPEGImages/*.jpg
│   └── SegmentationClass/*.png
├── NLP/Train/train-v2.0.json
├── Object-Detection/                    # reuses Image-Classification test2017
├── Speech-Recognition/*.wav
├── Super-Resolution/*.jpg
├── Text-Classification/WordVec/IMDB_Dataset.csv
└── Video-Classification/kinetics_600_5000/*.mp4

MODELBASE/
├── Autocomplete/*.tflite
├── Image-Classification/*.tflite
├── Image-Segmentation/*.tflite
├── NLP/*.tflite
├── Object-Detection/*.tflite
├── Speech-Recognition/*.tflite
├── Super-Resolution/*.tflite
├── Text-Classification/*.tflite
└── Video-Classification/*.tflite

LABELMAPS/
├── Image-Classification/labels.txt
├── Object-Detection/labelmap.txt
├── Speech-Recognition/conv_actions_labels.txt
├── Text-Classification/                   # if needed
└── Video-Classification/kinetics_600_labels.txt
```

### Python dependencies

```bash
pip install tensorflow jtop transformers datasets pandas opencv-python-headless soundfile matplotlib pillow numpy
playwright install chromium  # for browser workloads
```

### External tools (Non-AI workloads)

```
stress-ng fio sysbench dd ping ffmpeg openssl gzip make sqlite3
```

---

## Usage

### Standalone runner (CLI)

Each runner in `CODEBASE/<Category>/` accepts these arguments:

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--mode` | `-m` | `CPU1` | Execution mode: `CPU1` (1 thread), `CPU4` (4 threads), `GPU` |
| `--size` | `-s` | all | Limit number of samples to process |
| `--model` | | `None` | Filename of the `.tflite` model (without path) |
| `--delay` | `-d` | `0.5` | Delay between iterations in seconds |
| `--visualize` | `-v` | false | Show detection windows (Object Detection only) |

Example:
```bash
python CODEBASE/Image_Classification/img_classification_runner.py -m CPU4 -s 10 --model mobilenet -d 0.2
python CODEBASE/NLP/BERTQA.py -m GPU --model bert_qa --size 5
```

### Non-AI workloads

```bash
python Non_AI_runner.py -m CPU1 --model "Stress CPU" -s 3 --delay 0.5
```

Available models for Non-AI: `Stress CPU`, `FIO Disk`, `Sysbench CPU`, `DD Write`, `Ping Network`, `FFmpeg Encode`, `OpenSSL Crypto`, `Gzip Compression`, `Make Build`, `SQLite Ops`, `Web Browsing`.

### Run all benchmarks

```bash
python CODEBASE/test.py  # runs every *.py in CODEBASE/ with --mode and --size args
```

### Socket server

```bash
python server.py
```

Listens on port 50007. Send JSON:
```json
{"cmd": "run", "app_id": 1, "size": 10, "mode": "CPU4", "model": "mobilenet", "delay": 0.5}
```

Response:
```json
{"status": "finished", "duration": 1.234}
```

### App ID mapping

| ID | Application |
|----|------------|
| 0 | Autocomplete |
| 1 | Image Classification |
| 2 | Image Segmentation |
| 3 | Object Detection |
| 4 | NLP |
| 5 | Speech Recognition |
| 6 | Super Resolution |
| 7 | Text Classification |
| 8 | Video Classification |
| 9 | Non_AI |

---

## Output structure

Energy data (jtop-based):
```
logs/<Application>/<model>_<mode>.csv
```

Inference timings:
```
logs/<timestamp>/<Application>/<model>/<mode>/inference_timings.csv
```

Columns: `cycle, start_iso, end_iso, inference_s, total_s`

- `inference_s` — time spent in model invocation only
- `total_s` — full cycle including preprocessing/postprocessing
- For Non-AI and runners without preprocessing, both columns are equal
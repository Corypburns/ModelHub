# ModelHub

Energy fingerprinting system for AI and Non-AI workloads on Jetson edge devices.

## Entry points

- `server.py` — Socket server (port 50007). Receives `{"cmd": "run", "app_id", "size", "mode", "model"}` JSON messages and dispatches runs with energy monitoring.
- `CODEBASE/test.py` — Standalone benchmark runner. Executes all `*.py` files in `CODEBASE/` recursively with `--mode` and `--size` args.

## Key patterns

- All model runners accept CLI args: `-m/--mode` (CPU1 | CPU4 | GPU), `-s/--size`, `-v/--visualize`, `--model`.
- `server.py:run_ai()` wraps each run in an `EnergyMonitor` thread (jtop-based) that writes CSV to `logs/{app_name}/{model}_{mode}.csv`.
- `helper_functions.py:EnergyMonitor` requires **jtop** (Jetson Stats library) — will not work on non-Jetson hardware.
- `browser_load.py` uses Playwright headless Chromium; requires `playwright install chromium`.
- Non-AI workloads in `Non_AI_runner.py` run external commands (stress-ng, fio, sysbench, dd, ping, ffmpeg, openssl, gzip, make, sqlite3).

## Paths (from `CODEBASE/config.py`)

Relative to `Path.cwd()`: `DATASETS/`, `MODELBASE/`, `LABELMAPS/`, `logs/`.

## Directory ownership

`CODEBASE/Autocomplete/`, `CODEBASE/Image_Classification/`, `CODEBASE/Image_Segmentation/`, `CODEBASE/NLP/`, `CODEBASE/Object_Detection/`, `CODEBASE/Speech_Recognition/`, `CODEBASE/Super_Resolution/`, `CODEBASE/Text_Classification/`, `CODEBASE/Video_Classification/` — each contains a `run()` function used by `server.py:id_to_app`.
import subprocess
import os
from enum import Enum

from CODEBASE.helper_functions import get_base_parser
from browser_load import run as run_browser_load


# -----------------------------
# Non-AI Application Runners (10 total)
# -----------------------------

def run_stress_cpu(size=1, inference_timer=None, **kwargs):
    inference_timer.start_cycle() if inference_timer else None
    duration = size * 2
    subprocess.run(["stress-ng", "--cpu", "4", "--timeout", f"{duration}s"], check=True)
    inference_timer.end_cycle() if inference_timer else None


def run_fio_disk(size=1, inference_timer=None, **kwargs):
    inference_timer.start_cycle() if inference_timer else None
    filename = "fio_test_file"
    subprocess.run([
        "fio",
        "--name=randwrite",
        f"--size={size}G",
        f"--filename={filename}",
        "--rw=randwrite",
        "--bs=4k",
        "--iodepth=4",
        "--runtime=10",
        "--time_based"
    ], check=True)
    if os.path.exists(filename):
        os.remove(filename)
    inference_timer.end_cycle() if inference_timer else None


def run_sysbench_cpu(size=1, inference_timer=None, **kwargs):
    inference_timer.start_cycle() if inference_timer else None
    subprocess.run([
        "sysbench",
        "cpu",
        f"--cpu-max-prime={2500 * size}",
        "run"
    ], check=True)
    inference_timer.end_cycle() if inference_timer else None


def run_dd_write(size=1, inference_timer=None, **kwargs):
    inference_timer.start_cycle() if inference_timer else None
    filename = "dd_test_file"
    subprocess.run([
        "dd",
        "if=/dev/zero",
        f"of={filename}",
        "bs=1M",
        f"count={size * 1024}"
    ], check=True)
    if os.path.exists(filename):
        os.remove(filename)
    inference_timer.end_cycle() if inference_timer else None


def run_ping(size=1, inference_timer=None, **kwargs):
    inference_timer.start_cycle() if inference_timer else None
    subprocess.run(["ping", "-c", str(size * 3), "8.8.8.8"], check=True)
    inference_timer.end_cycle() if inference_timer else None


def run_ffmpeg(size=1, inference_timer=None, **kwargs):
    inference_timer.start_cycle() if inference_timer else None
    output = "test.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "testsrc=size=1280x720:rate=30",
        "-t", str(size * 2),
        "-c:v", "libx264",
        output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(output):
        os.remove(output)
    inference_timer.end_cycle() if inference_timer else None


def run_openssl(size=1, inference_timer=None, **kwargs):
    inference_timer.start_cycle() if inference_timer else None
    subprocess.run([
        "openssl", "speed",
        "-seconds", str(size)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    inference_timer.end_cycle() if inference_timer else None


def run_gzip(size=1, inference_timer=None, **kwargs):
    inference_timer.start_cycle() if inference_timer else None
    filename = "gzip_test"
    with open(filename, "wb") as f:
        f.write(os.urandom( 10 * 1024 *  1024 * size))
    subprocess.run(["gzip", "-k", filename], check=True)
    for f in [filename, filename + ".gz"]:
        if os.path.exists(f):
            os.remove(f)
    inference_timer.end_cycle() if inference_timer else None


def run_make(size=1, inference_timer=None, **kwargs):
    inference_timer.start_cycle() if inference_timer else None
    subprocess.run(["make", "-j", str(1000 * size)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    inference_timer.end_cycle() if inference_timer else None


def run_sqlite(size=1, inference_timer=None, **kwargs):
    inference_timer.start_cycle() if inference_timer else None
    db = "test.db"
    subprocess.run(["sqlite3", db, "CREATE TABLE t (id INT, val TEXT);"])
    for i in range(100 * size):
        subprocess.run(["sqlite3", db, f"INSERT INTO t VALUES({i}, 'data');"], stdout=subprocess.DEVNULL)
    if os.path.exists(db):
        os.remove(db)
    inference_timer.end_cycle() if inference_timer else None


class AppType(Enum):
    STRESS_CPU = run_stress_cpu
    FIO_DISK = run_fio_disk
    SYSBENCH_CPU = run_sysbench_cpu
    DD_WRITE = run_dd_write
    PING = run_ping
    FFMPEG = run_ffmpeg
    OPENSSL = run_openssl
    GZIP = run_gzip
    MAKE = run_make
    SQLITE = run_sqlite
    WEB = run_browser_load



run_map = {
    "Stress CPU": {"name": "Stress CPU", "run": AppType.STRESS_CPU},
    "FIO Disk": {"name": "FIO Disk", "run": AppType.FIO_DISK},
    "Sysbench CPU": {"name": "Sysbench CPU", "run": AppType.SYSBENCH_CPU},
    "DD Write": {"name": "DD Write", "run": AppType.DD_WRITE},
    "Ping Network": {"name": "Ping Network", "run": AppType.PING},
    "FFmpeg Encode": {"name": "FFmpeg Encode", "run": AppType.FFMPEG},
    "OpenSSL Crypto": {"name": "OpenSSL Crypto", "run": AppType.OPENSSL},
    "Gzip Compression": {"name": "Gzip Compression", "run": AppType.GZIP},
    "Make Build": {"name": "Make Build", "run": AppType.MAKE},
    "SQLite Ops": {"name": "SQLite Ops", "run": AppType.SQLITE},
    "Web Browsing": {"name": "Web Browsing", "run": AppType.WEB}
}


# -----------------------------
# Runner Logic
# -----------------------------
def run(mode, model, size, delay=0.5, inference_timer=None):
    app = run_map[model]
    print(f'Running {app["name"]}')
    if model == "Web Browsing":
        run_browser_load(size=size, delay=delay, inference_timer=inference_timer)
    else:
        app['run'](size, delay=delay, inference_timer=inference_timer)
    print(f'Finished with {app["name"]}')
    if inference_timer:
        inference_timer.flush()


def main():
    parser = get_base_parser('Run Non AI applications')
    args = parser.parse_args()

    run(mode=args.mode, model=args.model, size=args.size, delay=args.delay)

if __name__ == "__main__":
    main()
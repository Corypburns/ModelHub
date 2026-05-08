import subprocess
import os
from enum import Enum

from CODEBASE.helper_functions import get_base_parser
from browser_load import run as run_browser_load


# -----------------------------
# Non-AI Application Runners (10 total)
# -----------------------------

def run_stress_cpu(size=1, **kwargs):
    duration = size
    subprocess.run(["stress-ng", "--cpu", "4", "--timeout", f"{duration}s"], check=True)


def run_fio_disk(size=1, **kwargs):
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


def run_sysbench_cpu(size=1, **kwargs):
    subprocess.run([
        "sysbench",
        "cpu",
        f"--cpu-max-prime={2500 * size}",
        "run"
    ], check=True)


def run_dd_write(size=1, **kwargs):
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


def run_ping(size=1, **kwargs):
    subprocess.run(["ping", "-c", str(size * 3), "8.8.8.8"], check=True)


def run_ffmpeg(size=1, **kwargs):
    # Generate synthetic video and encode
    output = "test.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "testsrc=size=1280x720:rate=30",
        "-t", str(size),
        "-c:v", "libx264",
        output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(output):
        os.remove(output)


def run_openssl(size=1, **kwargs):
    subprocess.run([
        "openssl", "speed",
        "-seconds", str(size / 2)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_gzip(size=1, **kwargs):
    filename = "gzip_test"
    with open(filename, "wb") as f:
        f.write(os.urandom( 5 * 1024 *  1024 * size))
    subprocess.run(["gzip", "-k", filename], check=True)
    for f in [filename, filename + ".gz"]:
        if os.path.exists(f):
            os.remove(f)


def run_make(size=1, **kwargs):
    # Simulate compilation workload
    subprocess.run(["make", "-j", str(250 * size)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_sqlite(size=1, **kwargs):
    db = "test.db"
    subprocess.run(["sqlite3", db, "CREATE TABLE t (id INT, val TEXT);"]) 
    for i in range(100 * size):
        subprocess.run(["sqlite3", db, f"INSERT INTO t VALUES({i}, 'data');"], stdout=subprocess.DEVNULL)
    if os.path.exists(db):
        os.remove(db)


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
def run(mode, model, size):
    app = run_map[model]
    print(f'Running {app["name"]}')
    app['run'](size)
    print(f'Finished with {app["name"]}')


def main():
    parser = get_base_parser('Run Non AI applications')
    args = parser.parse_args()

    run(mode=args.mode, model=args.model, size=args.size)

if __name__ == "__main__":
    main()
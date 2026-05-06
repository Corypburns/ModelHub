
import argparse
import csv
import os
import threading
import time as t
import time
import numpy as np
from dataclasses import asdict, dataclass, field
import numpy as np
import tensorflow as tf
from .config import *

def load_model(model_path, num_threads):

    logging.info(f"\nLoading model: {model_path}")

    start_load = t.time()
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=num_threads
    )
    end_load = t.time()

    start_alloc = t.time()
    interpreter.allocate_tensors()
    end_alloc = t.time()

    logging.info(f"Model load time: {(end_load-start_load)*1000:.2f} ms")
    logging.info(f"Tensor allocation time: {(end_alloc-start_alloc)*1000:.2f} ms")

    return interpreter


def get_base_parser(description="TFLite Inference Script"):
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument(
        "-m", "--mode", 
        choices=["CPU1", "CPU4", "GPU"], 
        default="CPU1",
        help="Execution mode: CPU (1 thread), CPU (4 threads), or GPU"
    )
    
    parser.add_argument(
        "-s", "--size", 
        type=int, 
        default=None, 
        help="Limit the number of samples to process (default: all)"
    )

    parser.add_argument(
        "-v", "--visualize", 
        action="store_true", 
        help="Enable visualization window"
    )
    parser.add_argument(
        '--model', 
        default='None',
        help="File name of the specific model to run"
    )
    
    return parser


@dataclass
class Measurement:
    """
    Represents a set of electrical measurements at a given time.

    Attributes:
        t (float): Time of the measurement in seconds.
        power (float): Power in milliwatts (mW).
        max_v (float): Maximum voltage in millivolts (mV) across all rails.
        mean_v (float): Mean voltage in millivolts (mV) across all rails.
        max_c (float): Maximum current in milliamperes (mA) across all rails.
        mean_c (float): Mean current in milliamperes (mA) across all rails.
        ram_used (float): RAM used at the time of measurement (MB).
        ram_total (float): Total RAM available (MB).

    Methods:
        _set_voltage_current_stats(jt):
            Extracts voltage and current statistics (max and mean) from a measurement object.

        _set_total_power(jt):
            Extracts total power consumption from a measurement object.

        measure(jt, t):
            Updates the measurement instance with statistics from the given measurement object and time.

        __str__():
            Returns a formatted string representation of the measurement.

        __repr__():
            Returns a detailed string representation of the measurement for debugging.
    """

    t: float = 0.0
    power: float = 0.0
    max_v: float = 0.0
    mean_v: float = 0.0
    max_c: float = 0.0
    mean_c: float = 0.0
    ram_used: float = 0.0
    ram_total: float = 0.0

    def _set_voltage_current_stats(self, jt):
        rails = jt.power.get("rail", {}).values()
        vs, cs = [], []
        for r in rails:
            try:
                vs.append(float(r.get("volt", 0)))
                cs.append(float(r.get("curr", 0)))
            except:
                pass

        self.max_v = max(vs) if vs else 0
        self.mean_v = sum(vs) / len(vs) if vs else 0
        self.max_c = max(cs) if cs else 0
        self.mean_c = sum(cs) / len(cs) if cs else 0

    def _set_total_power(self, jt):
        p = jt.power
        for k in ("tot", "total", "Total"):
            if k in p:
                v = p[k]

                self.power = float(v["power"] if isinstance(v, dict) else v)
                return
        if "rail" in p:
            vals = [
                float(r.get("power", 0))
                for r in p["rail"].values()
                if isinstance(r, dict)
            ]
            self.power = sum(vals) / len(vals) if vals else 0.0

            return

    def _set_memory(self, jt):
        memory = jt.memory
        ram = memory.get("RAM", None)
        if not ram:
            raise Exception("Error trying to access RAM")
        m_used = ram.get("used", None)
        m_total = ram.get("tot", None)
        if not m_used or not m_total:
            raise Exception("Error trying to access total/used memory")
        self.ram_used = m_used / 1000
        self.ram_total = m_total / 1000

    def measure(self, jt, t):
        self._set_voltage_current_stats(jt)
        self._set_total_power(jt)
        self._set_memory(jt)
        self.t = t

    def __str__(self):
        return (
            f"Measurement(t={self.t:.6f}, power={self.power:.6f}, max_v={self.max_v:.6f}, "
            f"mean_v={self.mean_v:.6f}, max_c={self.max_c:.6f}, mean_c={self.mean_c:.6f}, "
            f"ram_used={self.ram_used:.6f}, ram_total={self.ram_total:.6f}"
        )

    def __repr__(self):
        return (
            f"Measurement(t={self.t!r}, power={self.power!r}, max_v={self.max_v!r}, "
            f"mean_v={self.mean_v!r}, max_c={self.max_c!r}, mean_c={self.mean_c!r}, "
            f"ram_used={self.ram_used!r}, ram_total={self.ram_total!r})"
        )


@dataclass
class Measurements:
    """
    Represents a set of measurements taken over a period of time.

    Attributes:
        lat (float): The latency between the first and last measurement (s).
        e_mj (float): The energy consumed during the measurement period (mJ).
        power (float): The average power consumption during the measurement period (mW).
        measurements (list[Measurement]): A list of individual measurement objects.
        max_v (float): The maximum voltage recorded across all measurements (mV).
        mean_v (float): The average voltage recorded across all measurements (mV).
        max_c (float): The maximum current recorded across all measurements (mA).
        mean_c (float): The average current recorded across all measurements (mA).
        ram_usage (float): The average memory usage (%).
    """

    lat: float = 0.0
    e_mj: float = 0.0
    power: float = 0.0
    ram_usage: float = 0.0
    max_v: float = 0.0
    mean_v: float = 0.0
    max_c: float = 0.0
    mean_c: float = 0.0

    measurements: list["Measurement"] = field(default_factory=list)

    def calc(self):
        if len(self.measurements) < 2:
            raise Exception("Not enough measure points")

        m0, m1 = self.measurements[0], self.measurements[1]
        self.lat = m1.t - m0.t

        mean_vs = [m.mean_v for m in self.measurements]
        mean_cs = [m.mean_c for m in self.measurements]
        max_vs = [m.max_v for m in self.measurements]
        max_cs = [m.max_c for m in self.measurements]
        total_p = [m.power for m in self.measurements]
        m_us = [m.ram_used for m in self.measurements]

        self.power = float(np.mean(total_p))
        self.e_mj = self.power * self.lat
        self.ram_usage = float(np.mean(m_us))
        self.max_v = max(max_vs)
        self.mean_v = float(np.mean(mean_vs))
        self.max_c = max(max_cs)
        self.mean_c = float(np.mean(mean_cs))

    def add_measurement(self, t, jt):
        if len(self.measurements) > 2:
            raise Exception("You can only have two measurement points")
        m = Measurement()
        m.measure(jt, t)
        self.measurements.append(m)

    def get_measurements(self):
        return [
            f"{self.lat  * 1000:.6f}",
            f"{self.e_mj:.6f}",
            f"{self.power:.6f}",
            f"{self.ram_usage:.6f}",
            f"{self.max_v:.6f}",
            f"{self.mean_v:.6f}",
            f"{self.max_c:.6f}",
            f"{self.mean_c:.6f}",
        ]

    def __str__(self):
        return (
            f"measurements(lat={self.lat:.6f}, e_mj={self.e_mj:.6f}, "
            f"power={self.power:.6f}, max_v={self.max_v:.6f}, mean_v={self.mean_v:.6f}, "
            f"max_c={self.max_c:.6f}, mean_c={self.mean_c:.6f}, ram_usage={self.ram_usage:.6f})"
        )

    def __repr__(self):
        return (
            f"measurements(lat={self.lat!r}, e_mj={self.e_mj!r}, power={self.power!r}, "
            f"max_v={self.max_v!r}, mean_v={self.mean_v!r}, max_c={self.max_c!r}, mean_c={self.mean_c!r}, ram_usage={self.ram_usage!r})"
        )


def avg_measurement(measurements: list[Measurements]):
    """
    Calculates the average and maximum values of various measurement attributes from a list of measurement objects.
    Args:
        measurements (list[Measurements]): List of measurement objects to process.

    Returns:
        tuple:
            (
                float: Average latency,
                float: Average energy in mJ,
                float: Average power,
                float: Average RAM usage,
                float: Maximum voltage,
                float: Average voltage,
                float: Maximum current,
                float: Average current
            )
        If the input list is empty, returns an empty tuple.
    """
    if not measurements:
        return {}
    for m in measurements:
        m.calc()

    m_array = np.array(
        [
            [m.lat, m.e_mj, m.power, m.ram_usage, m.max_v, m.mean_v, m.max_c, m.mean_c]
            for m in measurements
        ]
    )

    means = np.mean(m_array, axis=0)
    maxes = np.maximum.reduce(m_array[:, [4, 6]])  # max_v and max_c columns

    return (
        float(means[0]),  # lat
        float(means[1]),  # e_mJ
        float(means[2]),  # power
        float(means[3]),  # ram_usage
        float(maxes[0]),  # max_v
        float(means[4]),  # mean_v
        float(maxes[1]),  # max_c
        float(means[6]),  # mean_c
    )

MAX_RETRIES = 5
RETRY_DELAY = 0.3

class EnergyMonitor(threading.Thread):
    def __init__(self, jt, interval=0.1, output_file="energy_data.csv"):
        """
        Args:
            jt: The jtop/Jetson Stats object.
            interval (float): Seconds to wait between samples.
            output_file (str): Path to save the CSV.
        """
        super().__init__()
        self.jt = jt
        self.interval = interval
        self.output_file = output_file
        self.data = []
        self._stop_event = threading.Event()
        self.start_time = None

    def stop(self):
        """Signals the thread to stop collecting data."""
        self._stop_event.set()

    def run(self):
        logging.info("Energy monitoring started.")
        self.start_time = time.perf_counter()
        retry_count = 0
        while retry_count < MAX_RETRIES:
            while self.jt.ok():
                if self._stop_event.is_set():
                    break
                current_time = time.perf_counter() - self.start_time
                try:
                    m = Measurement()
                    m.measure(self.jt, current_time)
                    self.data.append(m)
                except Exception as e:
                    logging.error(f"Sampling error: {e}")

            if self._stop_event.is_set():
                break

            retry_count += 1
            if retry_count < MAX_RETRIES:
                logging.warning(f"jtop: Connection lost. Retry {retry_count}/{MAX_RETRIES} in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

        logging.info("Energy monitoring stopped. Saving data...")
        self._save_to_csv()

    def _save_to_csv(self):
        """Converts the list of dataclasses to a CSV file."""
        if not self.data:
            logging.warning("No data collected to save.")
            return

        try:
            # Get headers from the dataclass fields
            keys = asdict(self.data[0]).keys()
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                # Convert each Measurement object to a dict and write
                dict_writer.writerows([asdict(m) for m in self.data])
                
            logging.info(f"Data successfully saved to {self.output_file}")
        except Exception as e:
            logging.error(f"Failed to save CSV: {e}")
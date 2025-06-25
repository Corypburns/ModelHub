from jtop import jtop
import time

# Initialize jtop
with jtop() as jetson:
    while jetson.ok():
        stats = jetson.stats
        ram_used = stats['RAM']['used']
        ram_total = stats['RAM']['tot']
        print(f"RAM Usage: {ram_used:.2f} / {ram_total:.2f} MB")

        # optional sleep to reduce update frequency
        time.sleep(1.0)


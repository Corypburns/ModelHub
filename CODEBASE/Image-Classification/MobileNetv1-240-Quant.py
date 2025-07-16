import os
from datetime import datetime as dt

LOG_DIR = "E:/Code/Python/ModelHub/OUTPUTS/Image-Classification/MobileNetV1"
DATE_TIME = dt.now().strftime("%y%m%d_%H%M%S")
FILE_NAME = f"log_mobilenetv1-240_{DATE_TIME}.csv"
OUTPUT_PATH = os.path.join(LOG_DIR, FILE_NAME)
HEADERS = (
    "Timestamp,Review,Mode,"
    "Pre_Lat_ms,Inf_Lat_ms,Post_Lat_ms,"
    "Pre_E_mJ,Inf_E_mJ,Post_E_mJ,"
    "Pre_Max_V,Pre_Mean_V,Pre_Max_C,Pre_Mean_C,"
    "Inf_Max_V,Inf_Mean_V,Inf_Max_C,Inf_Mean_C,"
    "Post_Max_V,Post_Mean_V,Post_Max_C,Post_Mean_C,"
    "Pre_Pwr_mW,Inf_Pwr_mW,Post_Pwr_mW\n"
)

def init_csv():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'w') as f:
            f.write(HEADERS)

def append_csv_row(
        timestamp, review, mode,
        pre_lat_ms, inf_lat_ms, post_lat_ms,
        pre_e_mJ, inf_e_mJ, post_e_mJ,
        pre_max_v, pre_mean_v, pre_max_c, pre_mean_c,
        inf_max_v, inf_mean_v, inf_max_c, inf_mean_c,
        post_max_v, post_mean_v, post_max_c, post_mean_c,
        pre_pwr, inf_pwr, post_pwr      
):
    row = ",".join([
        timestamp,
        review,
        mode,
        f"{pre_lat_ms:.1f}", f"{inf_lat_ms:.1f}", f"{post_lat_ms:.1f}",
        f"{pre_e_mJ:.1f}", f"{inf_e_mJ:.1f}", f"{post_e_mJ:.1f}",
        f"{pre_max_v:.2f}", f"{pre_mean_v:.2f}", f"{pre_max_c:.2f}", f"{pre_mean_c:.2f}",
        f"{inf_max_v:.2f}", f"{inf_mean_v:.2f}", f"{inf_max_c:.2f}", f"{inf_mean_c:.2f}",
        f"{post_max_v:.2f}", f"{post_mean_v:.2f}", f"{post_max_c:.2f}", f"{post_mean_c:.2f}",
        f"{pre_pwr:.2f}", f"{inf_pwr:.2f}", f"{post_pwr:.2f}"
    ]) + "\n"
    with open(OUTPUT_PATH, 'a') as f:
        f.write(row)

def main():
    init_csv()
    append_csv_row(
        timestamp=DATE_TIME,
        review="test",
        mode="inference",
        pre_lat_ms=1.2, inf_lat_ms=2.3, post_lat_ms=0.9,
        pre_e_mJ=10.1, inf_e_mJ=20.2, post_e_mJ=5.5,
        pre_max_v=3.3, pre_mean_v=3.0, pre_max_c=0.5, pre_mean_c=0.45,
        inf_max_v=3.4, inf_mean_v=3.1, inf_max_c=0.6, inf_mean_c=0.55,
        post_max_v=3.2, post_mean_v=3.0, post_max_c=0.4, post_mean_c=0.35,
        pre_pwr=100.0, inf_pwr=200.0, post_pwr=80.0
    )

main()
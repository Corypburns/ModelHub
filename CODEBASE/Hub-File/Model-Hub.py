from pathlib import Path
import platform
import subprocess

if platform.system() == "Windows":
    base_path = Path(r"E:\Code\Python\ModelHub\CODEBASE")
else:
    base_path = Path.home()
    
location = None
choice = None
image_segmentation = base_path / "Deeplab_v3.py"

# === MODEL METHODS FOR MODELS WITH MORE THAN ONE SUBFOLDER/FILE ===
def object_detection_models():
    print(" 1) MobileNetV2-300x300 (FLOAT)\n",
          "2) MobileNetV2-300x300 (QUANTIZED)\n",
          "3) MobileNetV2-640x640 (FLOAT)\n",
          "4) MobileNetV2-640x640 (QUANTIZED)\n"
    )
    choice = int(input("Select a category -> "))
    
    match choice:
        case 1:
            location = base_path / "Object-Detection" /"MobileNetV2-300x300" / "MobileNetV2-300x300.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 2:
            location = base_path / "Object-Detection" / "MobileNetV2-300x300_Quantized" / "MobileNetV2-300x300_Quantized.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 3:
            location = base_path / "Object-Detection" / "MobileNetV2-640x640" / "MobileNetV2-640x640.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 4:
            location = base_path / "Object-Detection" / "MobileNetV2-640x640" / "MobileNetV2-640x640.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case _:
            print("Invalid entry.")
            return
def text_classification_models():
    print(" 1) WordVec\n",
          "2) MobileBert\n",
    )
    choice = int(input("Select a category -> "))
    
    match choice:
        case 1:
            location = base_path / "Text-Classification" /"WordVec" / "WordVec.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 2:
            location = base_path / "Text-Classification" /"MobileBert" / "MobileBert.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case _:
            print("Invalid entry.")
            return


def models_menu():
    print(" 1) Image-Segmentation\n",
          "2) Natural Language Processing\n",
          "3) Object Detection\n",
          "4) Super-Resolution\n",
          "5) Text-Classification\n")
    try:
        choice = int(input("Select a category -> "))
    except ValueError:
        print("Not a valid input")
        return
    
    
    match choice:
        case 1:
            location = base_path / "Image-Segmentation" / "Deeplab_v3.py"
            if location.exists():
                subprocess.run(["python", location])
            else:
                print("File does not exist or something is wrong.")
        case 2:
            location = base_path / "NLP" / "BERTQA.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError("File Not found")
        case 3:
            object_detection_models()
        case 4:
            location = base_path / "Super-Resolution" / "ESRGAN_TF.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 5:
            text_classification_models()
        case _:
            print("Invalid entry.")
            return


models_menu()
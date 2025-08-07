from pathlib import Path
import subprocess
import socket

host = socket.gethostname()

match host:
    case "DESKTOP-FI8GT7F":
        BASE_PATH = Path(r"E:\Code\Python\ModelHub\CODEBASE")
    case "CoryPC":
        BASE_PATH = None # Placeholder value for my laptop
    case "Placeholder": # Enter the host name of the computer you are working on
        None
    
location = None
choice = None

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
            location = BASE_PATH / "Object-Detection" /"MobileNetV2-300x300" / "MobileNetV2-300x300.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 2:
            location = BASE_PATH / "Object-Detection" / "MobileNetV2-300x300_Quantized" / "MobileNetV2-300x300_Quantized.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 3:
            location = BASE_PATH / "Object-Detection" / "MobileNetV2-640x640" / "MobileNetV2-640x640.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 4:
            location = BASE_PATH / "Object-Detection" / "MobileNetV2-640x640" / "MobileNetV2-640x640.py"
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
            location = BASE_PATH / "Text-Classification" /"WordVec" / "WordVec.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 2:
            location = BASE_PATH / "Text-Classification" /"MobileBert" / "MobileBert.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case _:
            print("Invalid entry.")
            return
def image_classification_models():
    print(" 1) EfficientNet_lite4 (FLOAT)\n",
          "2) EfficientNet_lite4 (QUANTIZED)\n",
          "3) MobileNetV1_224x224 (FLOAT)\n",
          "4) MobileNetV1-224x224 (QUANTIZED)\n",
          "5) InceptionV3"
    )
    choice = int(input("Select a category -> "))
    
    match choice:
        case 1:
            location = BASE_PATH / "Image-Classification" /"EfficientNet_lite4-224x224" / "EfficientNet_lite4.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 2:
            location = BASE_PATH / "Image-Classification" /"EfficientNet_lite4-224x224_Quantized" / "EfficientNet_lite4-224x224_Quantized.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 3:
            location = BASE_PATH / "Image-Classification" /"MobileNetV1-224x224" / "MobileNetV1-224x224.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 4:
            location = BASE_PATH / "Image-Classification" /"MobileNetV1-224x224_Quantized" / "MobileNetV1-224x224_Quantized.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 5:
            try:
                location = BASE_PATH / "Image-Classification" / "InceptionV3-299x299" / "InceptionV3-299x299.py"
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
          "5) Text-Classification\n",
          "6) Image-Classification\n")
    try:
        choice = int(input("Select a category -> "))
    except ValueError:
        print("Not a valid input")
        return
    
    
    match choice:
        case 1:
            location = BASE_PATH / "Image-Segmentation" / "Deeplab_v3.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 2:
            location = BASE_PATH / "NLP" / "BERTQA.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError("File Not found")
        case 3:
            object_detection_models()
        case 4:
            location = BASE_PATH / "Super-Resolution" / "ESRGAN_TF.py"
            try:
                subprocess.run(["python", location])
            except:
                FileNotFoundError
        case 5:
            text_classification_models()
        case 6:
            image_classification_models()
        case _:
            print("Invalid entry.")
            return


models_menu()
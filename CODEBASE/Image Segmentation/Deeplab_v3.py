import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# === CONFIG ===
VOC_DIR = r"E:\Code\Python\DATASETS\VOC2012_train_val"
MODEL_PATH = r"E:\Code\Python\ModelHub\MODELBASE\Image-Segmentation\deeplab_v3.tflite"

# === Load the TFLite model ===
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

# === Load validation image IDs ===
with open(os.path.join(VOC_DIR, "ImageSets", "Segmentation", "val.txt")) as f:
    val_ids = f.read().splitlines()

# === Define VOC color map ===
def label_to_color_image(label):
    colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
        [128, 192, 0], [0, 64, 128]
    ], dtype=np.uint8)
    return colors[label]

# === Loop over all images ===
for i, sample_id in enumerate(val_ids):
    print(f"Processing {i + 1}/{len(val_ids)}: {sample_id}")

    image_path = os.path.join(VOC_DIR, "JPEGImages", f"{sample_id}.jpg")
    mask_path = os.path.join(VOC_DIR, "SegmentationClass", f"{sample_id}.png")

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"Missing file: {sample_id}. Skipping.")
        continue

    # Load and preprocess the input image
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((input_width, input_height))
    input_image = np.array(resized_image, dtype=np.float32)
    input_image = input_image / 127.5 - 1.0
    input_image = np.expand_dims(input_image, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    segmentation_map = np.argmax(output_data, axis=-1)

    # Load and process ground truth
    gt_mask = Image.open(mask_path).resize((input_width, input_height), resample=Image.NEAREST)
    gt_array = np.array(gt_mask)
    gt_array[gt_array == 255] = 0  # Replace ignore index with 0
    gt_color = label_to_color_image(gt_array)

    # Convert prediction to color
    segmentation_color = label_to_color_image(segmentation_map)

    # Display result
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmentation_color)
    plt.title("Predicted Segmentation")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gt_color)
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=False)  # Non-blocking display
    plt.pause(3)           # Show for 3 seconds
    plt.close()

import os
import random
import cv2
import numpy as np
from rembg import remove
import onnxruntime

SOURCE_DIR = ""
OUTPUT_DIR = ""

def ensure_output_dir():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def remove_background_u2net(image_path):

    with open(image_path, "rb") as f:
        segmented_bytes = remove(f.read())

    nparr = np.frombuffer(segmented_bytes, np.uint8)
    segmented = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    alpha = segmented[:, :, 3] / 255.0
    rgb = segmented[:, :, :3]

    black_bg = np.zeros_like(rgb)

    alpha_3c = np.repeat(alpha[:, :, None], 3, axis=2)
    result = (rgb * alpha_3c + black_bg * (1 - alpha_3c)).astype(np.uint8)

    return result

def main():
    ensure_output_dir()

    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(".png")]

    for f in files:
        input_path = os.path.join(SOURCE_DIR, f)
        output_path = os.path.join(OUTPUT_DIR, f)

        print(f"Chosen file: {input_path}")

        result = remove_background_u2net(input_path)
        cv2.imwrite(output_path, result)

        print(f"Saved result to: {output_path}")

if __name__ == "__main__":
    main()


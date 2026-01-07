from skimage import exposure, io, img_as_ubyte, transform
import numpy as np
import os

# Replace with appropriate directories
SOURCE_DIR = ""
OUTPUT_DIR = ""

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SIZE = (512, 400)

def resize_with_aspect_ratio(img, target_size):
    target_h, target_w = target_size
    h, w = img.shape[:2]

    scale = min(target_h / h, target_w / w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    img_resized = transform.resize(
        img, (new_h, new_w), anti_aliasing=True, preserve_range=True
    )

    canvas = np.zeros((target_h, target_w, 3), dtype=img_resized.dtype)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_resized

    return canvas

files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(".png")]

for f in files:
    input_path = os.path.join(SOURCE_DIR, f)
    output_path = os.path.join(OUTPUT_DIR, f)

    img = io.imread(input_path).astype("float32") / 255.0

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img_resized = resize_with_aspect_ratio(img, TARGET_SIZE)

    img_clahe = img_resized.copy()
    for c in range(3):
        img_clahe[:, :, c] = exposure.equalize_adapthist(
            img_resized[:, :, c], clip_limit=0.01
        )

    # Adjust gamma value as needed
    img_gamma = exposure.adjust_gamma(img_clahe, gamma=0.7)

    img_uint8 = img_as_ubyte(img_gamma)
    io.imsave(output_path, img_uint8)

    print(f"Saved result to: {output_path}")

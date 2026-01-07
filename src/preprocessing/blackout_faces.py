import os
import cv2
import numpy as np

SOURCE_DIR = ""
OUTPUT_DIR = ""

PROTO_PATH = "../models/face_detector/deploy.prototxt"
MODEL_PATH = "../models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

CONF_THRESHOLD = 0.6

def ensure_output_dir():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_face_net():
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    return net

def detect_faces(net, image):
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )

    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < CONF_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return boxes

def blackout_face_region_ellipse(img, box, padding_ratio=0.15):
    (h, w) = img.shape[:2]
    (x1, y1, x2, y2) = box

    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * padding_ratio)
    pad_h = int(box_h * padding_ratio)

    x1p = max(0, x1 - pad_w)
    y1p = max(0, y1 - pad_h)
    x2p = min(w - 1, x2 + pad_w)
    y2p = min(h - 1, y2 + pad_h)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    center_x = (x1p + x2p) // 2
    center_y = (y1p + y2p) // 2
    axis_x = (x2p - x1p) // 2
    axis_y = (y2p - y1p) // 2

    cv2.ellipse(
        mask,
        (center_x, center_y),
        (axis_x, axis_y),
        0,
        0,
        360,
        255,
        thickness=-1,
    )

    img[mask == 255] = [0, 0, 0]

    return img

def main():
    ensure_output_dir()
    net = load_face_net()

    files = [
        f
        for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for fname in files:
        input_path = os.path.join(SOURCE_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, fname)

        img = cv2.imread(input_path)

        if img is None:
            continue

        boxes = detect_faces(net, img)

        if not boxes:
        else:
            for box in boxes:
                img = blackout_face_region_ellipse(img, box)

        cv2.imwrite(output_path, img)
        print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()

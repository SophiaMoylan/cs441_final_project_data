import os
import csv
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

# load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print(f"Loaded CLIP on {device}")

# file paths - change as needed
CHARACTERS = ["carrie", "charlotte", "miranda", "samantha"]

BASE_DIR = Path("data")
CSV_PATH = BASE_DIR / "all_outfits.csv"
OUTPUT_PATH = BASE_DIR / "clip_embeddings.jsonl"

def load_text_descriptions(csv_path):
    mapping = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]

            text = (
                f"{row['Short_Description']}. "
                f"Style aesthetic: {row['Style_Aesthetic']}. "
                f"Formality level: {row['Formality_Level']}."
            ).strip()

            mapping[filename] = text

    return mapping

@torch.no_grad()
def encode_image(path):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    img_emb = model.get_image_features(**inputs)
    img_emb = img_emb / img_emb.norm(p=2)
    return img_emb.squeeze().cpu().tolist()


@torch.no_grad()
def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt").to(device)
    txt_emb = model.get_text_features(**inputs)
    txt_emb = txt_emb / txt_emb.norm(p=2)
    return txt_emb.squeeze().cpu().tolist()

def main():
    text_map = load_text_descriptions(CSV_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:

        for character in CHARACTERS:
            folder = BASE_DIR / f"{character}_face_blackout_normalized2"

            if not folder.exists():
                continue

            image_files = [
                f for f in os.listdir(folder)
                if f.lower().endswith(".png")
            ]

            print(f"\nProcessing {character} ({len(image_files)} images)…")

            for fname in tqdm(image_files):
                img_path = folder / fname

                if fname not in text_map:
                    continue

                description = text_map[fname]

                img_emb = encode_image(img_path)
                txt_emb = encode_text(description)

                entry = {
                    "character": character,
                    "filename": fname,
                    "text": description,
                    "image_embedding": img_emb,
                    "text_embedding": txt_emb,
                }

                out_f.write(json.dumps(entry) + "\n")

    print(f"\n✓ Finished: embeddings saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


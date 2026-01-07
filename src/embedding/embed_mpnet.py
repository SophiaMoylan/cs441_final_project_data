import json
import torch
from transformers import AutoTokenizer, AutoModel

# load MPNet
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# insert as needed!
STYLE_DESCRIPTIONS = {
    "carrie": """
    ...
    """,
    "charlotte": """
    ...
    """,
    "miranda": """
    ...
    """,
    "samantha": """
    ...
    """
}

@torch.no_grad()
def encode_text(text: str):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)

    outputs = model(**tokens)

    token_embeddings = outputs.last_hidden_state
    attention_mask = tokens.attention_mask.unsqueeze(-1)

    summed = torch.sum(token_embeddings * attention_mask, dim=1)
    counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)

    emb = summed / counts
    emb = emb / emb.norm(p=2, dim=1, keepdim=True)

    return emb.squeeze().cpu().tolist()

def main():
    out_path = "character_style_embeddings.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for character, text in STYLE_DESCRIPTIONS.items():
            print(f"Embedding style description for {character}...")

            emb = encode_text(text)

            record = {
                "character": character,
                "description": text,
                "text_embedding": emb
            }

            f.write(json.dumps(record) + "\n")

    print(f"\nSaved style embeddings to {out_path}")


if __name__ == "__main__":
    main()
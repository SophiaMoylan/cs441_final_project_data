import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def load_style_embeddings(path):
    chars = []
    embeds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chars.append(rec["character"])
            embeds.append(np.array(rec["text_embedding"], dtype=np.float32))
    return chars, np.stack(embeds)

def load_clip_outfits(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

class OutfitDataset(Dataset):
    def __init__(self, outfits, chars):
        self.X = []
        self.Y = []

        char_to_idx = {c: i for i, c in enumerate(chars)}

        for rec in outfits:
            img_emb = np.array(rec["image_embedding"], dtype=np.float32)
            txt_emb = np.array(rec["text_embedding"], dtype=np.float32)
            x = np.concatenate([img_emb, txt_emb])
            y = char_to_idx[rec["character"]]
            self.X.append(x)
            self.Y.append(y)

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class StyleMLP(nn.Module):
    def __init__(self, input_dim, out_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        x = self.norm(x)
        z = self.net(x)
        return nn.functional.normalize(z, p=2, dim=1)
    
def train(model, train_loader, val_loader, style_embs, device="cpu", epochs=10):
    style_embs = torch.tensor(style_embs, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    model.to(device)

    for ep in range(epochs):
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optim.zero_grad()

            z = model(X)
            sims = z @ style_embs.T
            loss = loss_fn(sims, y)

            loss.backward()
            optim.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                z = model(X)
                sims = z @ style_embs.T
                loss = loss_fn(sims, y)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)

        print(f"Epoch {ep+1}/{epochs}  Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")


def show_incorrect_examples(result, chars, max_show=10):
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    fnames = result["filenames"]

    print("\n==============================")
    print(" Incorrect Predictions (sample)")
    print("==============================")

    wrong_idx = np.where(y_true != y_pred)[0]

    if len(wrong_idx) == 0:
        print("No misclassifications!")
        return

    for i in wrong_idx[:max_show]:
        t = chars[y_true[i]]
        p = chars[y_pred[i]]
        f = fnames[i]
        print(f"- {f}:  true={t}, predicted={p}")

def evaluate(model, outfits, chars, style_embs, device="cpu"):
    model.eval()

    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    y_true, y_pred = [], []
    filenames = []

    style_embs = torch.tensor(style_embs, dtype=torch.float32).to(device)

    for rec in outfits:
        img = np.array(rec["image_embedding"], dtype=np.float32)
        txt = np.array(rec["text_embedding"], dtype=np.float32)

        filenames.append(rec.get("filename", rec.get("id", "<no_name>")))

        x = torch.tensor(np.concatenate([img, txt])[None, :], dtype=torch.float32).to(device)

        with torch.no_grad():
            z = model(x)
            sims = (z @ style_embs.T).cpu().numpy()[0]

        pred_idx = int(np.argmax(sims))
        true_idx = char_to_idx[rec["character"]]

        y_pred.append(pred_idx)
        y_true.append(true_idx)

    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(chars))), zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(chars))))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "filenames": filenames,
    }

def main():
    style_path = "../data/character_style_embeddings.jsonl"
    clip_path = "../data/clip_embeddings.jsonl"

    chars, style_embs = load_style_embeddings(style_path)
    outfits = load_clip_outfits(clip_path)

    from collections import Counter
    print("Total outfit samples:", len(outfits))
    for c, n in Counter([rec["character"] for rec in outfits]).items():
        print(f"  {c}: {n}")

    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    labels = [rec["character"] for rec in outfits]
    fold = 1

    acc_list = []
    prec_all = []
    rec_all = []
    f1_all = []

    sample = outfits[0]
    input_dim = len(sample["image_embedding"]) + len(sample["text_embedding"])

    for train_idx, val_idx in skf.split(outfits, labels):
        print(f"\n==============================")
        print(f"Fold {fold}/{K}")
        print(f"==============================")

        train_outfits = [outfits[i] for i in train_idx]
        val_outfits   = [outfits[i] for i in val_idx]

        train_ds = OutfitDataset(train_outfits, chars)
        val_ds   = OutfitDataset(val_outfits, chars)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = StyleMLP(input_dim, out_dim=style_embs.shape[1])

        train(model, train_loader, val_loader, style_embs, device=device, epochs=10)

        result = evaluate(model, val_outfits, chars, style_embs, device=device)

        if fold == 5:
            show_incorrect_examples(result, chars, max_show=8)

        acc_list.append(result["accuracy"])
        prec_all.append(result["precision"])
        rec_all.append(result["recall"])
        f1_all.append(result["f1"])

        fold += 1

    prec_all = np.stack(prec_all)
    rec_all  = np.stack(rec_all)
    f1_all   = np.stack(f1_all)

    mean_prec = prec_all.mean(axis=0)
    std_prec  = prec_all.std(axis=0)

    mean_rec  = rec_all.mean(axis=0)
    std_rec   = rec_all.std(axis=0)

    mean_f1   = f1_all.mean(axis=0)
    std_f1    = f1_all.std(axis=0)

    macro_precisions = prec_all.mean(axis=1)
    macro_recalls    = rec_all.mean(axis=1)
    macro_f1s        = f1_all.mean(axis=1)

    mean_macro_prec = macro_precisions.mean()
    std_macro_prec  = macro_precisions.std()

    mean_macro_rec = macro_recalls.mean()
    std_macro_rec  = macro_recalls.std()

    mean_macro_f1 = macro_f1s.mean()
    std_macro_f1  = macro_f1s.std()

    mean_acc = np.mean(acc_list)
    std_acc  = np.std(acc_list)

    print("====================================")
    print("      CROSS-VALIDATION SUMMARY")
    print("====================================")
    print(f"Accuracy:        {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Macro Precision: {mean_macro_prec:.4f} ± {std_macro_prec:.4f}")
    print(f"Macro Recall:    {mean_macro_rec:.4f} ± {std_macro_rec:.4f}")
    print(f"Macro F1:        {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print("====================================\n")

    print("====================================")
    print("     PER-CLASS METRICS (AGGREGATED)")
    print("====================================")
    for i, c in enumerate(chars):
        print(
            f"{c:10s}  "
            f"Precision: {mean_prec[i]:.4f} ± {std_prec[i]:.4f}   "
            f"Recall: {mean_rec[i]:.4f} ± {std_rec[i]:.4f}   "
            f"F1: {mean_f1[i]:.4f} ± {std_f1[i]:.4f}"
        )


if __name__ == "__main__":
    main()

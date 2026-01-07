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
            img = np.array(rec["image_embedding"], dtype=np.float32)
            txt = np.array(rec["text_embedding"], dtype=np.float32)
            x = np.concatenate([img, txt])
            y = char_to_idx[rec["character"]]
            self.X.append(x)
            self.Y.append(y)

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class LinearStyleProjector(nn.Module):
    def __init__(self, input_dim, style_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, style_dim)

    def forward(self, x):
        z = self.proj(x)
        return nn.functional.normalize(z, p=2, dim=1)


def train(model, train_loader, val_loader, style_embs, device="cpu", epochs=7):
    style_embs = torch.tensor(style_embs, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    for ep in range(epochs):
        model.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            z = model(X)
            sims = z @ style_embs.T
            loss = loss_fn(sims, y)
            loss.backward()
            optim.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                z = model(X)
                sims = z @ style_embs.T
                loss = loss_fn(sims, y)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {ep+1}/{epochs}  Train={avg_train:.4f}  Val={avg_val:.4f}")

def evaluate(model, outfits, chars, style_embs, device="cpu"):
    model.eval()
    char_to_idx = {c: i for i, c in enumerate(chars)}

    y_true = []
    y_pred = []

    style_embs = torch.tensor(style_embs, dtype=torch.float32).to(device)

    for rec in outfits:
        img = np.array(rec["image_embedding"], dtype=np.float32)
        txt = np.array(rec["text_embedding"], dtype=np.float32)
        x = torch.tensor(np.concatenate([img, txt])[None, :], dtype=torch.float32).to(device)

        with torch.no_grad():
            z = model(x)
            sims = (z @ style_embs.T).cpu().numpy()[0]

        y_pred.append(int(np.argmax(sims)))
        y_true.append(char_to_idx[rec["character"]])

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(chars))), zero_division=0
    )
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    cm = confusion_matrix(y_true, y_pred)

    return acc, precision, recall, f1, cm

def main():

    style_path = "../data/character_style_embeddings.jsonl" # change as needed
    clip_path = "../data/clip_embeddings.jsonl"

    chars, style_embs = load_style_embeddings(style_path)
    outfits = load_clip_outfits(clip_path)

    labels = [rec["character"] for rec in outfits]
    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    sample = outfits[0]
    input_dim = len(sample["image_embedding"]) + len(sample["text_embedding"])
    style_dim = style_embs.shape[1]

    acc_list = []
    prec_classes_all = []
    rec_classes_all = []
    f1_classes_all = []

    fold = 1

    for train_idx, val_idx in skf.split(outfits, labels):

        print(f"\n==============================")
        print(f"       Fold {fold}/{K} (Model C)")
        print(f"==============================")

        train_outfits = [outfits[i] for i in train_idx]
        val_outfits   = [outfits[i] for i in val_idx]

        train_ds = OutfitDataset(train_outfits, chars)
        val_ds   = OutfitDataset(val_outfits, chars)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

        model = LinearStyleProjector(input_dim, style_dim)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        train(model, train_loader, val_loader, style_embs, device=device, epochs=10)

        acc, prec, rec, f1, cm = evaluate(model, val_outfits, chars, style_embs, device)

        acc_list.append(acc)
        prec_classes_all.append(prec)
        rec_classes_all.append(rec)
        f1_classes_all.append(f1)

        fold += 1

    prec_classes_all = np.stack(prec_classes_all)
    rec_classes_all  = np.stack(rec_classes_all)
    f1_classes_all   = np.stack(f1_classes_all)

    mean_prec = prec_classes_all.mean(axis=0)
    std_prec  = prec_classes_all.std(axis=0)

    mean_rec  = rec_classes_all.mean(axis=0)
    std_rec   = rec_classes_all.std(axis=0)

    mean_f1   = f1_classes_all.mean(axis=0)
    std_f1    = f1_classes_all.std(axis=0)

    macro_precisions = prec_classes_all.mean(axis=1)
    macro_recalls    = rec_classes_all.mean(axis=1)
    macro_f1s        = f1_classes_all.mean(axis=1)

    mean_macro_prec = macro_precisions.mean()
    std_macro_prec  = macro_precisions.std()

    mean_macro_rec  = macro_recalls.mean()
    std_macro_rec   = macro_recalls.std()

    mean_macro_f1   = macro_f1s.mean()
    std_macro_f1    = macro_f1s.std()

    mean_acc = np.mean(acc_list)
    std_acc  = np.std(acc_list)

    print("====================================")
    print("      CROSS-VALIDATION SUMMARY")
    print("====================================")
    print(f"Accuracy:        {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Macro Precision: {mean_macro_prec:.4f} ± {std_macro_prec:.4f}")
    print(f"Macro Recall:    {mean_macro_rec:.4f} ± {std_macro_rec:.4f}")
    print(f"Macro F1:        {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print("====================================")


    print("====================================")
    print("     PER-CLASS METRICS (AGGREGATED) ")
    print("====================================")

    for i, c in enumerate(chars):
        print(f"{c:10s}  "
            f"Precision: {mean_prec[i]:.4f} ± {std_prec[i]:.4f}   "
            f"Recall: {mean_rec[i]:.4f} ± {std_rec[i]:.4f}   "
            f"F1: {mean_f1[i]:.4f} ± {std_f1[i]:.4f}")

if __name__ == "__main__":
    main()

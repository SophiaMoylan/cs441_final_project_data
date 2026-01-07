import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)

def load_clip_outfits(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def build_xy(outfits, chars):
    char_to_idx = {c: i for i, c in enumerate(chars)}

    X_list = []
    y_list = []

    for rec in outfits:
        img = np.array(rec["image_embedding"], dtype=np.float32)
        txt = np.array(rec["text_embedding"], dtype=np.float32)
        x = np.concatenate([img, txt])
        y = char_to_idx[rec["character"]]

        X_list.append(x)
        y_list.append(y)

    # Concat both embeddings
    return np.stack(X_list), np.array(y_list)

def evaluate_scikit_lr(X, y, chars):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []

    prec_classes_all = []
    rec_classes_all = []
    f1_classes_all = []

    fold = 1

    for train_idx, val_idx in skf.split(X, y):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = LogisticRegression(
            max_iter=5000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val,
            y_pred,
            labels=list(range(len(chars))),
            zero_division=0
        )

        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()

        print(f"Accuracy: {acc:.4f}")
        print("Per-class metrics:")
        for i, c in enumerate(chars):
            print(f"{c:10s} P={precision[i]:.3f}  R={recall[i]:.3f}  F1={f1[i]:.3f}")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_val, y_pred, labels=list(range(len(chars))))
        print(pd.DataFrame(cm, index=chars, columns=chars))

        acc_list.append(acc)
        prec_list.append(macro_precision)
        rec_list.append(macro_recall)
        f1_list.append(macro_f1)

        prec_classes_all.append(precision)
        rec_classes_all.append(recall)
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


    print("\n====================================")
    print("      CROSS-VALIDATION SUMMARY      ")
    print("====================================")
    print(f"Accuracy:        {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Macro Precision: {np.mean(prec_list):.4f} ± {np.std(prec_list):.4f}")
    print(f"Macro Recall:    {np.mean(rec_list):.4f} ± {np.std(rec_list):.4f}")
    print(f"Macro F1:        {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print("====================================\n")


    print("====================================")
    print(" PER-CLASS METRICS ACROSS 5 FOLDS")
    print("====================================")
    for i, c in enumerate(chars):
        print(f"{c}:")
        print(f"  Precision: {mean_prec[i]:.4f} ± {std_prec[i]:.4f}")
        print(f"  Recall:    {mean_rec[i]:.4f} ± {std_rec[i]:.4f}")
        print(f"  F1 Score:  {mean_f1[i]:.4f} ± {std_f1[i]:.4f}")
        print("")

def main():
    clip_path = "../data/clip_embeddings.jsonl" # change as needed
    outfits = load_clip_outfits(clip_path)

    chars = sorted(list({rec["character"] for rec in outfits}))
    X, y = build_xy(outfits, chars)

    evaluate_scikit_lr(X, y, chars)


if __name__ == "__main__":
    main()

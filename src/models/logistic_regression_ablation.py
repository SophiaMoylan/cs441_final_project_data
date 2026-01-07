import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd

def load_clip_outfits(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def build_xy(outfits, chars):
    char_to_idx = {c: i for i, c in enumerate(chars)}
    X, Y = [], []

    for rec in outfits:
        # Swap out for image as needed
        emb = rec["text_embedding"]
        X.append(np.array(emb, dtype=np.float32))
        Y.append(char_to_idx[rec["character"]])

    return np.stack(X), np.array(Y, dtype=np.int64)

def evaluate_classifier(model, X_val, y_val, chars):
    y_pred = model.predict(X_val)

    accuracy = np.mean(y_pred == y_val)

    precision_per_class, recall_per_class, f1_per_class, _ = \
        precision_recall_fscore_support(
            y_val,
            y_pred,
            labels=list(range(len(chars))),
            average=None,
            zero_division=0
        )

    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()

    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(chars))))

    print("\nAccuracy:", accuracy)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=chars, columns=chars))

    return (
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        precision_per_class,
        recall_per_class,
        f1_per_class,
    )


def main():
    MODE = "text" # or image

    clip_path = "../data/clip_embeddings.jsonl" # change as needed
    outfits = load_clip_outfits(clip_path)

    chars = sorted(list({rec["character"] for rec in outfits}))
    labels = [rec["character"] for rec in outfits]

    X, y = build_xy(outfits, chars)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    prec_classes_all = []
    rec_classes_all = []
    f1_classes_all = []

    fold = 1

    for train_idx, val_idx in skf.split(X, labels):
        print("\n==========================")
        print(f" Fold {fold}/5  (TEXT-only)")
        print("==========================")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

        model = LogisticRegression(
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs"
        )

        model.fit(X_train, y_train)

        (
            acc,
            mp,
            mr,
            mf1,
            prec_pc,
            rec_pc,
            f1_pc,
        ) = evaluate_classifier(model, X_val, y_val, chars)

        acc_list.append(acc)
        prec_list.append(mp)
        rec_list.append(mr)
        f1_list.append(mf1)

        prec_classes_all.append(prec_pc)
        rec_classes_all.append(rec_pc)
        f1_classes_all.append(f1_pc)

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
    print("   CROSS-VALIDATION SUMMARY (TEXT ONLY)")
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


if __name__ == "__main__":
    main()


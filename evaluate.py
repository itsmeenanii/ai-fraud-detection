"""
evaluate.py - measure how accurate the fraud detector is.

Usage:
    python evaluate.py --mode rules      # free, offline
    python evaluate.py --mode llm        # needs API key (results are cached)
    python evaluate.py --mode hybrid     # what the app uses
    python evaluate.py --mode all        # compare all three side by side

Reads test_data.csv (columns: text,label) and writes:
    evaluation_results.csv   per-message predictions
    evaluation_summary.json  metrics (shown inside the Streamlit app)
    confusion_matrix.png     chart for your report / PPT
"""
import argparse
import csv
import json
import time

import fraud_core as fc

LABELS = list(fc.LABELS)


def predict(mode, text):
    if mode == "rules":
        return fc.rules_classify(text)[0]
    if mode == "llm":
        return fc.llm_classify(text)[0]
    cls, _, _, source = fc.hybrid_classify(text)
    if source.startswith("rules ("):   # LLM failed, so this isn't a real hybrid result
        raise fc.LLMUnavailable(source)
    return cls


def metrics(y_true, y_pred):
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    matrix = {t: {p: 0 for p in LABELS + ["Unknown"]} for t in LABELS}
    for t, p in zip(y_true, y_pred):
        matrix[t][p if p in LABELS else "Unknown"] += 1

    per_class = {}
    for c in LABELS:
        tp = matrix[c][c]
        fp = sum(matrix[t][c] for t in LABELS if t != c)
        fn = sum(v for p, v in matrix[c].items() if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 3), "recall": round(rec, 3),
                        "f1": round(f1, 3), "support": tp + fn}

    # "Is it dangerous?" view: Fraud vs everything else
    fraud_missed = sum(v for p, v in matrix["Fraud"].items() if p != "Fraud")
    false_alarms = sum(matrix[t]["Fraud"] for t in ("Legit", "Spam"))
    return {
        "samples": n,
        "accuracy": round(correct / n, 3),
        "macro_f1": round(sum(v["f1"] for v in per_class.values()) / len(LABELS), 3),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "fraud_missed": fraud_missed,
        "false_fraud_alarms": false_alarms,
    }


def print_report(mode, m):
    print(f"\n=== {mode.upper()} ===  accuracy {m['accuracy']*100:.1f}%  "
          f"macro-F1 {m['macro_f1']:.3f}  ({m['samples']} samples)")
    print(f"{'':>8}  precision  recall    f1  support")
    for c in LABELS:
        s = m["per_class"][c]
        print(f"{c:>8}  {s['precision']:>9.2f}  {s['recall']:>6.2f}  {s['f1']:>4.2f}  {s['support']:>7}")
    print("\nConfusion matrix (rows = actual, cols = predicted)")
    print(f"{'':>8}" + "".join(f"{p:>9}" for p in LABELS + ['Unknown']))
    for t in LABELS:
        print(f"{t:>8}" + "".join(f"{m['confusion_matrix'][t][p]:>9}" for p in LABELS + ['Unknown']))
    print(f"\nFraud messages missed: {m['fraud_missed']}   |   "
          f"Non-fraud flagged as fraud: {m['false_fraud_alarms']}")


def plot_matrix(m, mode, path="confusion_matrix.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cols = LABELS + ["Unknown"]
    data = [[m["confusion_matrix"][t][p] for p in cols] for t in LABELS]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix ({mode}) - accuracy {m['accuracy']*100:.1f}%")
    top = max(max(r) for r in data) or 1
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center",
                    color="white" if v > top / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rules", "llm", "hybrid", "all"], default="rules")
    ap.add_argument("--data", default="test_data.csv")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r["text"] for r in rows]
    y_true = [r["label"] for r in rows]

    modes = ["rules", "llm", "hybrid"] if args.mode == "all" else [args.mode]
    summary, preds_by_mode = {}, {}

    for mode in modes:
        preds = []
        try:
            for t in texts:
                preds.append(predict(mode, t))
                if mode in ("llm", "hybrid"):
                    time.sleep(0.2)  # be gentle with rate limits
        except fc.LLMUnavailable as e:
            print(f"\n[{mode}] skipped - LLM unavailable: {e}")
            continue
        m = metrics(y_true, preds)
        print_report(mode, m)
        summary[mode] = m
        preds_by_mode[mode] = preds

    if not summary:
        return

    best = max(summary, key=lambda k: summary[k]["accuracy"])
    plot_matrix(summary[best], best)
    with open("evaluation_summary.json", "w") as f:
        json.dump({"best_mode": best, "modes": summary}, f, indent=2)

    with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "actual"] + [f"pred_{m}" for m in preds_by_mode])
        for i, t in enumerate(texts):
            w.writerow([t, y_true[i]] + [preds_by_mode[m][i] for m in preds_by_mode])

    print("\nSaved: evaluation_summary.json, evaluation_results.csv, confusion_matrix.png")


if __name__ == "__main__":
    main()

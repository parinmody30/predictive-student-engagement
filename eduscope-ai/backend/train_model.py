"""Train once, save fitted pipelines + measured metrics. Usage: python backend/train_model.py [--data CSV] [--out DIR]"""
import argparse, datetime, json, logging, sys
from pathlib import Path
import joblib, pandas as pd, sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from model_pipeline import LABELS, SEED, build_pipeline, feature_list, validate_df

BASE = Path(__file__).resolve().parent
log = logging.getLogger("train")


def evaluate(df, use_score, mode):
    feats = feature_list(use_score)
    if mode == "stratified":
        tr, te = train_test_split(df, test_size=0.2, stratify=df["final_result"], random_state=SEED)
        split, hold = "stratified random 80/20 (random_state=42)", None
    else:
        pres = sorted(df["code_presentation"].dropna().unique())
        if len(pres) < 2:
            return None, None
        hold = pres[-1]
        tr, te = df[df["code_presentation"] != hold], df[df["code_presentation"] == hold]
        split = f"course-presentation holdout: train on other presentations, test on {hold}"
    pipe = build_pipeline(feats)
    pipe.fit(tr[feats], tr["final_result"])  # fitted on the training part only
    pred = pipe.predict(te[feats])
    y = te["final_result"]
    rep = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
    return pipe, {
        "name": f"Random Forest ({'with' if use_score else 'without'} day-30 score)", "model_type": "RandomForestClassifier (fixed defaults, no search)",
        "split": split, "holdout_presentation": hold, "features": feats, "labels": LABELS, "train_size": len(tr), "test_size": len(te),
        "accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=LABELS, zero_division=0)),
        "per_class": {l: {"precision": rep[l]["precision"], "recall": rep[l]["recall"], "f1": rep[l]["f1-score"], "support": int(rep[l]["support"])} for l in LABELS},
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
        "test_class_distribution": {l: int((y == l).sum()) for l in LABELS}}


def train(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evals = []
    for key, use in (("primary", False), ("with_score", True)):
        pipe, e = evaluate(df, use, "stratified")
        e["key"] = key
        joblib.dump({"pipeline": pipe, "features": feature_list(use), "labels": LABELS, "uses_score": use,
                     "version": f"random-forest-day30-{key}-v1", "sklearn_version": sklearn.__version__}, out_dir / f"rf_day30_{key}.joblib")
        evals.append(e)
        _, h = evaluate(df, use, "holdout")
        if h:
            h["key"] = key
            evals.append(h)
    metrics = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "dataset_rows": len(df),
               "overall_class_distribution": {l: int((df["final_result"] == l).sum()) for l in LABELS}, "evaluations": evals}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "data" / "oulad_day30_features_audited.csv"))
    ap.add_argument("--out", default=str(BASE / "models"))
    a = ap.parse_args()
    if not Path(a.data).exists():
        sys.exit(f"Dataset not found: {a.data}\nCopy oulad_day30_features_audited.csv into backend/data/ or pass --data.")
    try:
        df, info = validate_df(pd.read_csv(a.data))
    except ValueError as e:
        sys.exit(f"Invalid dataset: {e}")
    log.info("Loaded %s rows, %s duplicate keys", info["rows"], info["dup"])
    m = train(df, a.out)
    for e in m["evaluations"]:
        print(f"{e['name']} | {e['split']} | test={e['test_size']} acc={e['accuracy']:.4f} macroF1={e['macro_f1']:.4f} weightedF1={e['weighted_f1']:.4f}")
    print("Saved models and metrics.json to", a.out)

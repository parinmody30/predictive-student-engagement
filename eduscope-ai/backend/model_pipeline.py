"""Shared feature definitions, dataset/record validation and pipeline construction."""
import os
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
LABELS = ["Distinction", "Fail", "Pass", "Withdrawn"]
REQUIRED_COLS = ["code_module", "code_presentation", "id_student", "gender", "region", "highest_education", "imd_band", "age_band",
                 "num_of_prev_attempts", "studied_credits", "disability", "days_before_start_registered", "clicks_by_day30",
                 "active_days_by_day30", "assessments_submitted_by_day30", "average_score_by_day30", "final_result"]
NUM = ["num_of_prev_attempts", "studied_credits", "days_before_start_registered", "clicks_by_day30",
       "active_days_by_day30", "assessments_submitted_by_day30"]
CAT = ["code_module", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]
SCORE = "average_score_by_day30"  # availability by day 30 is unverified -> optional, off in the primary model
TEXT = CAT + ["code_presentation", "final_result"]
NUMERIC_COLS = NUM + [SCORE, "id_student"]
# Not model inputs: final_result (target), id_student (identifier), code_presentation (cohort label; keeps holdout fair).


def feature_list(use_score):
    return CAT + NUM + ([SCORE] if use_score else [])


def validate_df(df):
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError("Missing required columns: " + ", ".join(miss))
    if df.empty:
        raise ValueError("The file has no rows.")
    df = df.copy()
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # unparseable -> NaN (never 0)
    for c in TEXT:
        df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip()).replace("", np.nan)
    bad = int((~df["final_result"].isin(LABELS)).sum())
    if bad:
        raise ValueError(f"{bad} rows have final_result outside {LABELS}.")
    mv = {c: int(n) for c, n in df.isna().sum().items() if n}
    dup = int(df.duplicated(["code_module", "code_presentation", "id_student"]).sum())
    return df, {"rows": len(df), "cols": len(df.columns), "dup": dup, "mv": mv, "bad": bad, "h": list(df.columns)}


def validate_record(rec, feats):
    if not isinstance(rec, dict):
        return None, ["record must be a JSON object"]
    out, errs = {}, []
    for f in feats:
        if f not in rec:
            errs.append(f"missing field: {f}")
            continue
        v = rec[f]
        if f in CAT:
            out[f] = np.nan if v is None or v == "" else str(v)
            continue
        if v is None or v == "":
            out[f] = np.nan  # missing stays missing; imputed inside the fitted pipeline
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            errs.append(f"{f} must be numeric")
            continue
        if not np.isfinite(x) or x < 0 or (f == SCORE and x > 100):
            errs.append(f"{f} is out of range")
            continue
        out[f] = x
    return out, errs


def to_frame(clean, feats):
    X = pd.DataFrame(clean, columns=feats)
    for c in CAT:
        if c in X:
            X[c] = X[c].astype(object)
    return X


def build_pipeline(feats, n_estimators=None, min_samples_leaf=None):
    # Defaults 300 trees / leaf 2. RF_TREES and RF_MIN_LEAF shrink the model for small hosts (see render.yaml).
    n_estimators = n_estimators or int(os.getenv("RF_TREES", 300))
    min_samples_leaf = min_samples_leaf or int(os.getenv("RF_MIN_LEAF", 2))
    num = [f for f in feats if f in NUM or f == SCORE]
    cat = [f for f in feats if f in CAT]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, class_weight="balanced_subsample",
                                random_state=SEED, n_jobs=-1)
    return Pipeline([("pre", pre), ("rf", rf)])

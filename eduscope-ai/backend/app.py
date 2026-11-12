"""EduScope AI Flask API. Run: python backend/app.py  -> http://localhost:5000"""
import hmac, io, json, logging, os
from pathlib import Path
import joblib, pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from model_pipeline import LABELS, to_frame, validate_df, validate_record
import numpy as np

log = logging.getLogger("eduscope")
BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
WARN_RULE = "Flagged when active_days_by_day30 is below the threshold (default 5). A screening signal for human review, not a validated probability or a model prediction."


def create_app(data_path=None, models_dir=None, admin_key=None):
    # ADMIN_KEY set (e.g. on Render): uploads need header X-Admin-Key. Unset: uploads open (local use only).
    admin_key = admin_key if admin_key is not None else os.getenv("ADMIN_KEY", "")
    data_path = Path(data_path or BASE / "data" / "oulad_day30_features_audited.csv")
    models_dir = Path(models_dir or BASE / "models")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024
    CORS(app)
    S = {"df": None, "info": None, "models": {}, "err": None}

    def load_data():
        S["df"] = S["info"] = S["err"] = None
        if data_path.exists():
            try:
                df, info = validate_df(pd.read_csv(data_path))
                info["name"] = data_path.name
                S["df"], S["info"] = df, info
            except Exception as e:
                S["err"] = str(e)
                log.error("Dataset invalid: %s", e)

    def load_models():
        for key in ("primary", "with_score"):
            p = models_dir / f"rf_day30_{key}.joblib"
            if p.exists():
                try:
                    S["models"][key] = joblib.load(p)
                    log.info("Loaded model %s", p.name)
                except Exception as e:
                    log.error("Could not load %s: %s", p, e)

    load_data()
    load_models()

    def err(msg, code=400, **kw):
        return jsonify({"error": msg, **kw}), code

    def js(df):
        return json.loads(df.to_json(orient="records"))

    def no_data():
        return err(S["err"] or "No dataset loaded. Upload oulad_day30_features_audited.csv.", 409)

    @app.errorhandler(Exception)
    def handle(e):
        if isinstance(e, HTTPException):
            return err(e.description, e.code)
        log.exception("Unhandled error")
        return err("Internal server error", 500)

    @app.get("/")
    def index():
        return send_from_directory(ROOT, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "upload_protected": bool(admin_key), "dataset_loaded": S["df"] is not None, "dataset_file": (S["info"] or {}).get("name"),
                        "dataset_error": S["err"], "models": [{"key": k, "version": m["version"]} for k, m in S["models"].items()]})

    @app.get("/api/dataset/summary")
    def summary():
        if S["df"] is None:
            return no_data()
        df = S["df"]
        return jsonify({"info": S["info"], "outcomes": {l: int((df.final_result == l).sum()) for l in LABELS},
                        "modules": df.code_module.value_counts().to_dict(), "presentations": df.code_presentation.value_counts().to_dict(),
                        "engagement_means": {c: None if pd.isna(df[c].mean()) else float(df[c].mean()) for c in
                                             ["clicks_by_day30", "active_days_by_day30", "assessments_submitted_by_day30", "average_score_by_day30"]}})

    @app.post("/api/dataset/upload")
    def upload():
        if admin_key and not hmac.compare_digest(request.headers.get("X-Admin-Key", "").encode(), admin_key.encode()):
            return err("Uploading requires the admin key.", 401)
        f = request.files.get("file")
        if not f or not f.filename:
            return err("Attach a CSV as form field 'file'.")
        if not f.filename.lower().endswith(".csv"):
            return err("Only .csv files are accepted.")
        raw = f.read()
        try:
            df, info = validate_df(pd.read_csv(io.BytesIO(raw)))
        except ValueError as e:
            return err(str(e))
        except Exception as e:
            return err(f"Could not parse CSV: {e}")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(raw)
        load_data()
        return jsonify({**info, "note": "Dataset replaced. Models are NOT retrained automatically; run train_model.py."})

    @app.get("/api/students")
    def students():
        if S["df"] is None:
            return no_data()
        a, df = request.args, S["df"]
        for col, arg in (("code_module", "module"), ("code_presentation", "presentation"), ("final_result", "outcome")):
            if a.get(arg):
                df = df[df[col] == a[arg]]
        q = a.get("q", "").strip()
        if q:
            df = df[df.id_student.map(lambda x: "" if pd.isna(x) else str(int(x))).str.contains(q, regex=False)]
        sort = a.get("sort", "id_student")
        if sort not in df.columns:
            return err(f"Unknown sort column: {sort}")
        df = df.sort_values(sort, ascending=a.get("order", "asc") != "desc", na_position="last")
        try:
            page, size = max(1, int(a.get("page", 1))), min(200, max(1, int(a.get("page_size", 25))))
        except ValueError:
            return err("page and page_size must be integers")
        return jsonify({"total": len(df), "page": page, "page_size": size, "records": js(df.iloc[(page - 1) * size: page * size])})

    @app.get("/api/students/all")
    def students_all():
        if S["df"] is None:
            return no_data()
        return jsonify({"info": S["info"], "records": js(S["df"])})

    def run_predictions(recs, use_score):
        m = S["models"].get("with_score" if use_score else "primary")
        if m is None:
            return None, [], "Model not loaded. Run: python backend/train_model.py"
        ok, errs = [], []
        for i, r in enumerate(recs):
            c, e = validate_record(r, m["features"])
            if e:
                errs.append({"index": i, "errors": e})
            else:
                ok.append((i, c))
        out = []
        if ok:
            pipe = m["pipeline"]
            P = pipe.predict_proba(to_frame([c for _, c in ok], m["features"]))
            cls = list(pipe.classes_)
            for (i, _), p in zip(ok, P):
                r = recs[i]
                out.append({"index": i, "student_id": r.get("id_student"), "course_module": r.get("code_module"),
                            "course_presentation": r.get("code_presentation"), "predicted_outcome": cls[int(np.argmax(p))],
                            "class_probabilities": {c: float(v) for c, v in zip(cls, p)},
                            "probability_note": "Model-estimated class probability; calibration not evaluated.",
                            "model_version": m["version"], "uses_score_feature": m["uses_score"],
                            "feature_cutoff": "course day 30", "prediction_mode": "model"})
        return out, errs, None

    @app.post("/api/predict")
    def predict():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return err("Request body must be a JSON object of features.")
        us = body.get("use_score", False)
        if not isinstance(us, bool):
            return err("use_score must be true or false")
        out, errs, msg = run_predictions([body], us)
        if msg:
            return err(msg, 503)
        if errs:
            return err("Invalid input", 422, details=errs[0]["errors"])
        return jsonify(out[0])

    @app.post("/api/predict/batch")
    def predict_batch():
        body = request.get_json(silent=True)
        recs = body.get("records") if isinstance(body, dict) else body
        us = body.get("use_score", False) if isinstance(body, dict) else False
        if not isinstance(recs, list) or not recs or not isinstance(us, bool):
            return err("Send {\"records\": [...], \"use_score\": false} with a non-empty list.")
        if len(recs) > 5000:
            return err("At most 5000 records per request.", 413)
        out, errs, msg = run_predictions(recs, us)
        if msg:
            return err(msg, 503)
        return jsonify({"results": out, "invalid": errs}), (200 if out else 422)

    @app.get("/api/model/metrics")
    def metrics():
        p = models_dir / "metrics.json"
        if not p.exists():
            return err("No measured metrics found. Run: python backend/train_model.py", 404)
        return jsonify(json.loads(p.read_text()))

    @app.get("/api/analytics/outcomes")
    def outcomes():
        if S["df"] is None:
            return no_data()
        df = S["df"]
        by = {m: {l: int(((g.final_result == l)).sum()) for l in LABELS} for m, g in df.groupby("code_module")}
        return jsonify({"total": len(df), "counts": {l: int((df.final_result == l).sum()) for l in LABELS}, "by_module": by})

    @app.get("/api/analytics/engagement")
    def engagement():
        if S["df"] is None:
            return no_data()
        cols = ["clicks_by_day30", "active_days_by_day30", "assessments_submitted_by_day30", "average_score_by_day30"]
        agg = lambda g: json.loads(g[cols].agg(["mean", "median"]).to_json())
        df = S["df"]
        return jsonify({"by_outcome": {k: agg(g) for k, g in df.groupby("final_result")}, "by_module": {k: agg(g) for k, g in df.groupby("code_module")}})

    @app.get("/api/early-warning")
    def early():
        if S["df"] is None:
            return no_data()
        try:
            thr, lim = float(request.args.get("active_days_threshold", 5)), int(request.args.get("limit", 200))
        except ValueError:
            return err("active_days_threshold and limit must be numbers")
        df = S["df"]
        if request.args.get("module"):
            df = df[df.code_module == request.args["module"]]
        d = df[df.active_days_by_day30 < thr].sort_values(["active_days_by_day30", "clicks_by_day30"])
        rows = []
        for r in js(d.head(lim)):
            why = [f"active days by day 30 ({r['active_days_by_day30']:g}) below threshold {thr:g}"]
            if r["assessments_submitted_by_day30"] == 0:
                why.append("no assessment submissions recorded by day 30")
            rows.append({**{k: r[k] for k in ["id_student", "code_module", "code_presentation", "clicks_by_day30", "active_days_by_day30", "assessments_submitted_by_day30", "average_score_by_day30"]}, "reasons": why})
        return jsonify({"rule": WARN_RULE, "type": "rule-based (not a model prediction)", "threshold": thr, "total_flagged": len(d), "records": rows})

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=False)

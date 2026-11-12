# EduScope AI: full-stack version

The original single-file frontend (`index.html`) is kept as is and now talks to a Flask + scikit-learn backend. Flask also serves the page, so one command runs everything.

## Windows setup (PowerShell)
```
cd eduscope-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
copy "C:\path\to\oulad_day30_features_audited.csv" backend\data\
python backend\train_model.py
python backend\app.py
```
Open http://localhost:5000. Run tests: `python -m unittest discover -s tests -v`.

You can also open `index.html` directly; it then calls http://localhost:5000 (change under Settings).

## Deploy to Render
`render.yaml` (repo root) is a Render Blueprint. It installs dependencies, trains the model during the build, and runs the app with gunicorn.
1. Commit the dataset: `eduscope-ai/backend/data/oulad_day30_features_audited.csv`. The build fails without it.
2. In Render: **New > Blueprint**, pick this repo and branch, then **Apply**. The site goes live at `https://eduscope-ai.onrender.com` (or a similar name).
3. `ADMIN_KEY` is generated for you. Find it under the service's **Environment** tab. You only need it to upload a replacement CSV from Data Management; visitors cannot upload without it.

Notes: on the free plan the server sleeps after ~15 min idle, so the first visit takes ~1 minute to wake it. The disk is temporary, so an uploaded CSV is lost on restart or redeploy. To change the data permanently, commit the new CSV and Render retrains it. The deployed forest uses `RF_TREES=150`, `RF_MIN_LEAF=10` to fit 512 MB RAM; the measured metrics shown in the app are for that model.

## What `train_model.py` does
Validates the CSV, trains Random Forests once (no search at startup), and writes `backend/models/*.joblib` and `metrics.json`, all measured on held-out data:
- **primary**: day-30 features *without* `average_score_by_day30` (score availability by day 30 is unverified).
- **with_score**: experimental variant including the score; missing scores stay NaN and are median-imputed inside the pipeline.
- Each is evaluated on a stratified 80/20 split (seed 42) and on a course-presentation holdout (latest presentation held out). Fixed settings: 300 trees, min_samples_leaf=2, balanced_subsample (override with env vars `RF_TREES`, `RF_MIN_LEAF`). These are not your tuned Colab parameters; edit `build_pipeline` to use them.
- Never inputs: `final_result`, `id_student`, and `code_presentation` (a cohort label that would undermine the holdout). `code_module`, demographics and day-30 engagement are used.
- Your Colab numbers (RF 52.8% / XGBoost 57.3%) are historical; the app shows the freshly measured ones. XGBoost is not required.

## Colab artifact
To use your own model, save `joblib.dump({"pipeline": pipe, "features": [...], "labels": [...], "uses_score": False, "version": "my-v1"}, "rf_day30_primary.joblib")` from Colab with the same scikit-learn version and copy it to `backend/models/`.

## API
| Endpoint | Purpose |
|---|---|
| GET /api/health | server, dataset, loaded models |
| GET /api/dataset/summary | counts, outcomes, engagement means |
| POST /api/dataset/upload | CSV (form field `file`), validated, replaces dataset |
| GET /api/students | `page, page_size, q, module, presentation, outcome, sort, order` |
| GET /api/students/all | all records (used by the frontend) |
| POST /api/predict | one record JSON (+ optional `use_score`) |
| POST /api/predict/batch | `{"records":[...],"use_score":false}` |
| GET /api/model/metrics | measured metrics, per-class, confusion matrices |
| GET /api/analytics/outcomes, /engagement | outcome and engagement summaries |
| GET /api/early-warning | rule-based flags: `active_days_threshold` (default 5), `module`, `limit` |

Sample request:
```
$b = @{code_module="AAA";gender="M";region="East Anglian Region";highest_education="HE Qualification";imd_band="20-30%";age_band="0-35";disability="N";num_of_prev_attempts=0;studied_credits=60;days_before_start_registered=30;clicks_by_day30=120;active_days_by_day30=9;assessments_submitted_by_day30=1} | ConvertTo-Json
Invoke-RestMethod http://localhost:5000/api/predict -Method Post -ContentType application/json -Body $b
```
Probabilities are model-estimated, not calibrated. The early warning rule is a documented screening threshold, not a model or validated probability.

## Troubleshooting
- "Backend: offline": start `python backend\app.py`; click the status button to retry.
- 503 "Model not loaded": run `train_model.py` first.
- 409 "No dataset": copy the CSV into `backend\data\` or upload it in Data Management.
- Model load error after copying from Colab: match the scikit-learn version.
- Port busy: `$env:PORT=5001; python backend\app.py` and update the API URL in Settings.

## Limitations
Random-split results do not prove generalization to new presentations (see holdout). Classes are imbalanced. Predictions support, not replace, advisor judgment. OULAD has no employment labels, so nothing here predicts employability.

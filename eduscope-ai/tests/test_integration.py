"""Integration tests. The synthetic rows below exist ONLY to exercise code paths; they are never shipped or shown as app data."""
import io, sys, tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from model_pipeline import LABELS, validate_df
import train_model
from app import create_app


def fixture(n=900, seed=1):
    r = np.random.default_rng(seed)
    act = r.integers(0, 31, n)
    res = np.where(act < 4, "Withdrawn", np.where(act < 12, "Fail", np.where(act < 24, "Pass", "Distinction"))).astype(object)
    flip = r.random(n) < .25
    res[flip] = r.choice(LABELS, flip.sum())
    return pd.DataFrame({"code_module": r.choice(list("ABC"), n), "code_presentation": r.choice(["2013J", "2014B", "2014J"], n),
        "id_student": np.arange(1000, 1000 + n), "gender": r.choice(["M", "F"], n), "region": r.choice(["East", "West"], n),
        "highest_education": r.choice(["A Level", "HE"], n), "imd_band": r.choice(["0-10%", "10-20%", None], n),
        "age_band": r.choice(["0-35", "35-55"], n), "num_of_prev_attempts": r.integers(0, 3, n), "studied_credits": r.choice([30, 60], n),
        "disability": r.choice(["N", "Y"], n), "days_before_start_registered": r.integers(0, 100, n),
        "clicks_by_day30": act * r.integers(5, 40, n), "active_days_by_day30": act, "assessments_submitted_by_day30": r.integers(0, 3, n),
        "average_score_by_day30": np.where(r.random(n) < .5, np.nan, r.integers(30, 100, n)), "final_result": res})


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.csv = cls.tmp / "d.csv"
        cls.df = fixture()
        cls.df.to_csv(cls.csv, index=False)
        train_model.train(validate_df(pd.read_csv(cls.csv))[0], cls.tmp / "m")
        cls.c = create_app(cls.csv, cls.tmp / "m").test_client()
        cls.rec = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in cls.df.iloc[0].items() if k != "final_result"}

    def test_health_and_models(self):
        j = self.c.get("/api/health").get_json()
        self.assertTrue(j["dataset_loaded"])
        self.assertEqual({m["key"] for m in j["models"]}, {"primary", "with_score"})

    def test_summary_counts(self):
        j = self.c.get("/api/dataset/summary").get_json()
        self.assertEqual(j["info"]["rows"], 900)
        self.assertEqual(sum(j["outcomes"].values()), 900)

    def test_predict_valid(self):
        r = self.c.post("/api/predict", json=self.rec)
        j = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertIn(j["predicted_outcome"], LABELS)
        self.assertAlmostEqual(sum(j["class_probabilities"].values()), 1, places=5)
        self.assertEqual(j["prediction_mode"], "model")

    def test_predict_with_missing_score(self):
        r = self.c.post("/api/predict", json={**self.rec, "average_score_by_day30": None, "use_score": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["uses_score_feature"])

    def test_predict_errors(self):
        bad = {k: v for k, v in self.rec.items() if k != "clicks_by_day30"}
        self.assertEqual(self.c.post("/api/predict", json=bad).status_code, 422)
        self.assertEqual(self.c.post("/api/predict", json={**self.rec, "clicks_by_day30": "abc"}).status_code, 422)
        self.assertEqual(self.c.post("/api/predict", data="x").status_code, 400)

    def test_batch(self):
        j = self.c.post("/api/predict/batch", json={"records": [self.rec, {"gender": "M"}]}).get_json()
        self.assertEqual(len(j["results"]), 1)
        self.assertEqual(j["invalid"][0]["index"], 1)

    def test_metrics_are_real(self):
        j = self.c.get("/api/model/metrics").get_json()
        e = [x for x in j["evaluations"] if x["key"] == "primary" and not x["holdout_presentation"]][0]
        self.assertEqual(e["test_size"], 180)
        self.assertNotIn("average_score_by_day30", e["features"])
        self.assertTrue(any(x["holdout_presentation"] for x in j["evaluations"]))

    def test_students_and_warning(self):
        j = self.c.get("/api/students?page_size=10&q=1000&outcome=Pass").get_json()
        self.assertLessEqual(len(j["records"]), 10)
        w = self.c.get("/api/early-warning?active_days_threshold=3").get_json()
        self.assertEqual(w["total_flagged"], int((self.df.active_days_by_day30 < 3).sum()))

    def test_upload_validation(self):
        f = lambda t: {"file": (io.BytesIO(t.encode()), "x.csv")}
        self.assertEqual(self.c.post("/api/dataset/upload", data=f("a,b\n1,2"), content_type="multipart/form-data").status_code, 400)
        self.assertEqual(self.c.post("/api/dataset/upload", data={}, content_type="multipart/form-data").status_code, 400)

    def test_no_dataset(self):
        c = create_app(self.tmp / "none.csv", self.tmp / "none").test_client()
        self.assertEqual(c.get("/api/students").status_code, 409)
        self.assertEqual(c.post("/api/predict", json=self.rec).status_code, 503)


if __name__ == "__main__":
    unittest.main()

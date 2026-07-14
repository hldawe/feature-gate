"""
Feature Gate Demo Runner with ML Validation
=============================================
Sends three real-world datasets to the Feature Gate API, prints screening
results, then builds scikit-learn models to validate the shortlist.

For each dataset, three models are trained and compared:
  1. SHORTLIST — only features Feature Gate flagged as significant
  2. ALL       — every feature included
  3. REJECTED  — only features Feature Gate filtered out

Usage:
    1. Start the server:  python main.py  (in the feature-gate directory)
    2. Run this script:   python run_demos.py

Requires: scikit-learn, requests, numpy, pandas
    pip install scikit-learn requests numpy pandas
"""

import json
import sys
import os
import time
import warnings

import numpy as np
import requests

API_URL = os.environ.get("FEATURE_GATE_URL", "http://localhost:8000")
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

DEMOS = [
    {
        "file": "demo1_tips_payload.json",
        "title": "Restaurant Tips",
        "question": "What drives how much a diner tips?",
        "source": "Bryant & Smith (1995) — real restaurant data, 244 meals",
        "expect": "total_bill should rank #1; party_size moderate; smoker/day/time likely noise",
    },
    {
        "file": "demo2_mpg_payload.json",
        "title": "Car Fuel Economy",
        "question": "What determines a car's miles per gallon?",
        "source": "UCI Auto MPG (StatLib/Carnegie Mellon) — real cars 1970-1973",
        "expect": "weight and displacement should rank highest; car_name (free text) is the wild card",
    },
    {
        "file": "demo3_diamonds_payload.json",
        "title": "Diamond Pricing",
        "question": "What determines how much a diamond costs?",
        "source": "Blue Nile / ggplot2 (Wickham) — 500 real diamonds, sampled from 53,940",
        "expect": "carat should dominate; cut/color/clarity should add value; depth/table likely noise",
    },
]


# ── Feature Gate screening output ─────────────────────────────────────────────

def print_results(result, demo):
    """Pretty-print screening results."""
    print(f"\n{'='*80}")
    print(f"  {demo['title'].upper()}")
    print(f"  {demo['question']}")
    print(f"  Source: {demo['source']}")
    print(f"{'='*80}")

    print(f"\n  Features screened: {result['n_features_screened']}")
    print(f"  Significant (q < {result['fdr_alpha']}): {result['n_significant']}")
    if result.get("n_with_structure"):
        print(f"  With temporal structure: {result['n_with_structure']}")
    if result.get("warnings"):
        for w in result["warnings"]:
            print(f"  ⚠ {w}")
    print()

    print(f"  {'Feature':<25} {'Type':<12} {'Cosine':>8} {'cos-p':>8} "
          f"{'dCor':>8} {'Lag':>5} {'Struct':>6} {'Combined-p':>11} {'q-value':>9} {'Score':>7}  Sig?")
    print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*6} {'-'*11} {'-'*9} {'-'*7}  ----")

    for f in result["results"]:
        sig = " ***" if f["significant"] else ""
        dcor = f"{f['dcor_statistic']:.4f}" if f.get("dcor_statistic") is not None else "    —"
        lag = f"{f['best_lag']:>4}τ" if f.get("best_lag") is not None else "    —"
        struct = "yes" if f.get("has_structure") else "—"
        name = f["name"] + ("*" if f.get("derived") else "")

        print(f"  {name:<25} {f['inferred_dtype']:<12} "
              f"{f['cosine_score']:>8.4f} {f['cosine_p_value']:>8.4f} "
              f"{dcor:>8} {lag:>5} {struct:>6} "
              f"{f['combined_p_value']:>11.6f} {f['q_value']:>9.6f} "
              f"{f['alignment_score']:>7.1f}{sig}")

    if any(f.get("derived") for f in result["results"]):
        print(f"\n  * = derived feature (Fourier expansion)")
    structured_not_sig = [f["name"] for f in result["results"]
                          if f.get("has_structure") and not f["significant"]]
    if structured_not_sig:
        print(f"  ⚠ Structure without significance (review recommended): {', '.join(structured_not_sig)}")

    print(f"\n  Expected: {demo['expect']}")
    print()


# ── ML validation ─────────────────────────────────────────────────────────────

def infer_column_type(values):
    """Determine if a column is numeric, categorical, or text."""
    numeric_count = 0
    total = 0
    for v in values:
        if v is None:
            continue
        total += 1
        try:
            float(v)
            numeric_count += 1
        except (ValueError, TypeError):
            pass

    if total == 0:
        return "categorical"
    if numeric_count / total > 0.8:
        return "numeric"

    str_vals = [str(v) for v in values if v is not None]
    avg_len = np.mean([len(s) for s in str_vals]) if str_vals else 0
    return "text" if avg_len > 50 else "categorical"


def build_feature_matrix(columns, n_rows):
    """
    Build a feature matrix from a list of column dicts using ColumnTransformer.

    Handles:
      - numeric:     pass through with median imputation
      - categorical: one-hot encoding
      - text:        TF-IDF (max 50 features)

    Returns (X_array, transformer) or (None, None) if no columns.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    import pandas as pd

    if not columns:
        return None, None

    numeric_cols = []
    categorical_cols = []
    text_cols = []

    df = pd.DataFrame()
    for col in columns:
        name = col["name"]
        dtype = col.get("dtype") or infer_column_type(col["values"])

        if dtype == "numeric":
            df[name] = [float(v) if v is not None else np.nan for v in col["values"]]
            numeric_cols.append(name)
        elif dtype == "text":
            df[name] = [str(v) if v is not None else "" for v in col["values"]]
            text_cols.append(name)
        else:
            df[name] = [str(v) if v is not None else "" for v in col["values"]]
            categorical_cols.append(name)

    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            categorical_cols,
        ))

    for tc in text_cols:
        transformers.append((
            f"text_{tc}",
            TfidfVectorizer(max_features=50, stop_words="english"),
            tc,
        ))

    if not transformers:
        return None, None

    ct = ColumnTransformer(transformers, remainder="drop")
    X = ct.fit_transform(df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    return X, ct


def run_ml_validation(payload, screening_result):
    """
    Build three GradientBoosting models and compare cross-validated R².

    Uses TimeSeriesSplit for time-indexed data, regular KFold otherwise.

    Returns dict with keys SHORTLIST, ALL, REJECTED, each containing
    r2_mean, r2_std, n_features_raw, n_features_encoded, folds.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold

    y_values = payload["y"]["values"]
    y = np.array([float(v) if v is not None else np.nan for v in y_values])
    valid_mask = ~np.isnan(y)
    y_clean = y[valid_mask]
    n_rows = len(y_values)

    # Detect if this is time-indexed data
    is_time_indexed = payload["y"].get("is_time_indexed", False)
    if not is_time_indexed:
        is_time_indexed = any(col.get("is_time_indexed", False) for col in payload["x_columns"])

    sig_names = {f["name"] for f in screening_result["results"] if f["significant"]}
    shortlisted = [col for col in payload["x_columns"] if col["name"] in sig_names]
    rejected = [col for col in payload["x_columns"] if col["name"] not in sig_names]

    warnings.filterwarnings("ignore", category=UserWarning)

    n_folds = 3 if len(y_clean) < 120 else 5
    if is_time_indexed:
        cv = TimeSeriesSplit(n_splits=n_folds)
        cv_label = f"TimeSeriesSplit({n_folds})"
        model_label = "Ridge"
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_label = f"{n_folds}-fold CV"
        model_label = "GradientBoosting"

    results = {"_cv_method": cv_label, "_model": model_label}
    configs = [
        ("SHORTLIST", shortlisted),
        ("ALL", payload["x_columns"]),
        ("REJECTED", rejected),
    ]

    for label, columns in configs:
        if not columns:
            results[label] = {
                "r2_mean": float("nan"), "r2_std": float("nan"),
                "n_features_raw": 0, "n_features_encoded": 0, "folds": 0,
            }
            continue

        X, ct = build_feature_matrix(columns, n_rows)
        if X is None:
            results[label] = {
                "r2_mean": float("nan"), "r2_std": float("nan"),
                "n_features_raw": len(columns), "n_features_encoded": 0, "folds": 0,
            }
            continue

        X_clean = X[valid_mask] if len(X) == len(valid_mask) else X

        # Ridge for time series (can extrapolate trends); GBR for cross-sectional
        if is_time_indexed:
            model = Ridge(alpha=1.0)
        else:
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
            )

        try:
            scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring="r2")
            results[label] = {
                "r2_mean": float(np.mean(scores)),
                "r2_std": float(np.std(scores)),
                "n_features_raw": len(columns),
                "n_features_encoded": X_clean.shape[1],
                "folds": n_folds,
            }
        except Exception as e:
            results[label] = {
                "r2_mean": float("nan"), "r2_std": float("nan"),
                "n_features_raw": len(columns), "n_features_encoded": X_clean.shape[1],
                "folds": n_folds, "error": str(e),
            }

    warnings.resetwarnings()
    return results


def print_ml_results(ml_results, screening_result):
    """Pretty-print the three-model comparison."""
    sig_names = [f["name"] for f in screening_result["results"] if f["significant"]]
    rej_names = [f["name"] for f in screening_result["results"] if not f["significant"]]

    cv_method = ml_results.get("_cv_method", "cross-validated")
    model_name = ml_results.get("_model", "GradientBoosting")
    print(f"\n  {'─'*72}")
    print(f"  ML VALIDATION ({model_name}, {cv_method} R²)")
    print(f"  {'─'*72}")
    print()

    print(f"  {'Model':<14} {'Raw Feat':<10} {'Encoded':<9} {'R² Mean':>9} {'± Std':>8}  Features Used")
    print(f"  {'-'*14} {'-'*10} {'-'*9} {'-'*9} {'-'*8}  {'-'*30}")

    for label in ["SHORTLIST", "ALL", "REJECTED"]:
        r = ml_results[label]
        n_raw = r.get("n_features_raw", 0)
        n_enc = r.get("n_features_encoded", 0)
        r2 = r.get("r2_mean", float("nan"))
        std = r.get("r2_std", float("nan"))

        if label == "SHORTLIST":
            names = ", ".join(sig_names) if sig_names else "(none)"
            marker = " ◀"
        elif label == "REJECTED":
            names = ", ".join(rej_names) if rej_names else "(none)"
            marker = ""
        else:
            names = "(all features)"
            marker = ""

        if np.isnan(r2):
            if r.get("error"):
                print(f"  {label:<14} {n_raw:<10} {'—':<9} {'ERROR':>9} {'':>8}  {names}")
            else:
                print(f"  {label:<14} {0:<10} {'—':<9} {'N/A':>9} {'':>8}  (no features)")
        else:
            print(f"  {label:<14} {n_raw:<10} {n_enc:<9} {r2:>9.4f} {std:>7.4f}  {names}{marker}")

    print()

    # Interpretation
    sl = ml_results["SHORTLIST"]
    al = ml_results["ALL"]
    rj = ml_results["REJECTED"]

    sl_r2 = sl.get("r2_mean", float("nan"))
    al_r2 = al.get("r2_mean", float("nan"))
    rj_r2 = rj.get("r2_mean", float("nan"))

    if np.isnan(sl_r2):
        return

    if not np.isnan(al_r2) and al_r2 > 0:
        if sl_r2 >= al_r2 * 0.95:
            print(f"  ✓ SHORTLIST matches ALL ({sl_r2:.3f} vs {al_r2:.3f})"
                  f" — Feature Gate found the signal")
        elif sl_r2 >= al_r2 * 0.80:
            pct_r2 = sl_r2 / al_r2 * 100
            print(f"  ~ SHORTLIST close to ALL ({sl_r2:.3f} vs {al_r2:.3f})"
                  f" — most signal captured ({pct_r2:.0f}%)")
        elif sl_r2 > 0:
            pct_r2 = sl_r2 / al_r2 * 100
            print(f"  ~ SHORTLIST below ALL ({sl_r2:.3f} vs {al_r2:.3f})"
                  f" — partial signal captured ({pct_r2:.0f}%)")
        else:
            print(f"  △ SHORTLIST R² is negative ({sl_r2:.3f} vs ALL {al_r2:.3f})"
                  f" — shortlisted features capture the dominant signal (trend) but the"
                  f" model needs additional features for accurate extrapolation")

    if not np.isnan(rj_r2):
        if rj_r2 < 0.1:
            print(f"  ✓ REJECTED has near-zero R² ({rj_r2:.3f})"
                  f" — correctly identified as noise")
        elif not np.isnan(sl_r2) and sl_r2 > 0 and rj_r2 < sl_r2 * 0.5:
            print(f"  ~ REJECTED has weak R² ({rj_r2:.3f})"
                  f" — mostly noise, minor residual signal")
        elif rj_r2 > 0.3:
            print(f"  △ REJECTED has R² ({rj_r2:.3f})"
                  f" — some encoded features carry signal (e.g. text tokens, one-hot categories)")
        else:
            print(f"  ✓ REJECTED is weak ({rj_r2:.3f})"
                  f" — limited signal on its own")

    if not np.isnan(al_r2) and al_r2 > 0 and sl_r2 > 0:
        print(f"  → Feature Gate selected {sl.get('n_features_raw', 0)}"
              f"/{al.get('n_features_raw', 0)} features"
              f" ({sl.get('n_features_raw', 0)/max(al.get('n_features_raw', 1), 1)*100:.0f}%)"
              f" capturing {sl_r2/max(al_r2, 0.001)*100:.0f}% of the R²")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Check sklearn availability
    try:
        import sklearn  # noqa: F401
        has_sklearn = True
    except ImportError:
        print("⚠ scikit-learn not installed. Run: pip install scikit-learn")
        print("  ML validation will be skipped.\n")
        has_sklearn = False

    # Health check
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        health = resp.json()
        if health.get("status") != "ok":
            print(f"Server unhealthy: {health}")
            sys.exit(1)
        print(f"✓ Feature Gate is running at {API_URL}")
        print(f"  Tier-1 loaded: {health.get('tier1_loaded', '?')}")
    except requests.ConnectionError:
        print(f"✗ Cannot connect to {API_URL}")
        print(f"  Start the server first: cd feature-gate && python main.py")
        sys.exit(1)

    # Run each demo
    for demo in DEMOS:
        path = os.path.join(DEMO_DIR, demo["file"])
        with open(path) as f:
            payload = json.load(f)

        n_features = len(payload["x_columns"])
        n_rows = len(payload["y"]["values"])
        print(f"\n▶ Sending: {demo['title']} ({n_rows} rows × {n_features} features)...")

        # Step 1: Feature Gate screening
        t0 = time.time()
        resp = requests.post(f"{API_URL}/screen", json=payload, timeout=600)
        gate_elapsed = time.time() - t0

        if resp.status_code != 200:
            print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
            continue

        result = resp.json()
        print_results(result, demo)
        print(f"  ⏱ Feature Gate completed in {gate_elapsed:.1f}s")

        # Step 2: ML validation
        if has_sklearn and result["n_significant"] > 0:
            print(f"\n  Building ML models to validate the shortlist...")
            t1 = time.time()
            ml_results = run_ml_validation(payload, result)
            ml_elapsed = time.time() - t1
            print_ml_results(ml_results, result)
            print(f"\n  ⏱ ML validation completed in {ml_elapsed:.1f}s")
        elif has_sklearn and result["n_significant"] == 0:
            print(f"\n  No significant features — skipping ML validation")
        print()


if __name__ == "__main__":
    main()

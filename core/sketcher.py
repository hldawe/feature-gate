"""
Advanced sketch construction (Section 3.1 of the paper).

Converts each column into a compact JSON->string summary encoding distributional
statistics, entropy, string-shape features, and optional temporal fingerprints.
"""
import json
import numpy as np
from typing import Any
from collections import Counter


def _numeric_sketch(
    values: np.ndarray,
    is_time_indexed: bool = False,
    acf_lags: list[int] | None = None,
) -> dict:
    """Build sketch dict for a numeric column."""
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return {"type": "numeric", "n": 0, "all_nan": True}

    sketch: dict[str, Any] = {
        "type": "numeric",
        "n": int(len(clean)),
        "min": float(np.min(clean)),
        "q25": float(np.percentile(clean, 25)),
        "median": float(np.median(clean)),
        "q75": float(np.percentile(clean, 75)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
        "mad": float(np.median(np.abs(clean - np.median(clean)))),
    }

    n = len(clean)
    if n >= 3:
        m = np.mean(clean)
        s = np.std(clean, ddof=1)
        if s > 0:
            sketch["skew"] = round(float(np.mean(((clean - m) / s) ** 3)), 4)
            if n >= 4:
                sketch["kurtosis"] = round(float(np.mean(((clean - m) / s) ** 4) - 3), 4)

    iqr = sketch["q75"] - sketch["q25"]
    if iqr > 0:
        lo, hi = sketch["q25"] - 1.5 * iqr, sketch["q75"] + 1.5 * iqr
        sketch["outlier_rate_lo"] = round(float(np.mean(clean < lo)), 4)
        sketch["outlier_rate_hi"] = round(float(np.mean(clean > hi)), 4)

    if is_time_indexed and acf_lags and n > max(acf_lags, default=0) + 1:
        acf_vals = {}
        centered = clean - np.mean(clean)
        var = np.dot(centered, centered)
        if var > 0:
            for lag in acf_lags:
                if lag < n:
                    acf_val = np.dot(centered[:n - lag], centered[lag:]) / var
                    acf_vals[f"acf_{lag}"] = round(float(acf_val), 4)
        sketch["acf"] = acf_vals

    return sketch


def _categorical_sketch(values: list[Any], top_k: int = 10) -> dict:
    """Build sketch dict for a categorical/text column."""
    str_vals = [str(v) for v in values if v is not None and str(v).strip() != ""]
    if len(str_vals) == 0:
        return {"type": "categorical", "n": 0, "all_empty": True}

    counts = Counter(str_vals)
    n_unique = len(counts)
    total = len(str_vals)
    top_items = counts.most_common(top_k)

    all_probs = np.array([c / total for c in counts.values()])
    entropy = -float(np.sum(all_probs * np.log2(all_probs + 1e-12)))

    lengths = [len(s) for s in str_vals]
    sample = str_vals[:500]
    digit_fracs = [sum(c.isdigit() for c in s) / max(len(s), 1) for s in sample]
    upper_fracs = [sum(c.isupper() for c in s) / max(len(s), 1) for s in sample]

    sketch: dict[str, Any] = {
        "type": "categorical",
        "n": total,
        "n_unique": n_unique,
        "top_values": [{"val": v[:80], "count": c} for v, c in top_items],
        "entropy": round(entropy, 4),
        "avg_length": round(float(np.mean(lengths)), 2),
        "digit_fraction": round(float(np.mean(digit_fracs)), 4),
        "upper_fraction": round(float(np.mean(upper_fracs)), 4),
    }
    return sketch


def _token_sketch(values: list[Any], top_k: int = 30) -> dict:
    """
    Token-level sketch for text columns (v1.1).

    Treats each row as a document. Tokenises (lowercase, alphanumeric),
    scores tokens by df * log(n_docs / df) — favouring tokens that recur
    across rows without being universal — and returns the top-k with counts.

    This gives the embedder semantic content to work with, addressing the
    limitation that purely statistical sketches cannot distinguish text
    columns that differ in meaning but not in shape.
    """
    import re
    docs = [str(v).lower() for v in values if v is not None and str(v).strip() != ""]
    n_docs = len(docs)
    if n_docs == 0:
        return {"tokens": []}

    # Document frequency: number of rows containing each token
    df_counts: Counter = Counter()
    total_counts: Counter = Counter()
    token_re = re.compile(r"[a-z0-9]+")
    for doc in docs:
        tokens = set()
        for tok in token_re.findall(doc):
            if len(tok) >= 2:  # skip single chars
                total_counts[tok] += 1
                tokens.add(tok)
        df_counts.update(tokens)

    if not df_counts:
        return {"tokens": []}

    # Score: df * log(n_docs / df). Zero for tokens in every row (no discriminative value)
    # and low for tokens in only one row (likely noise). Peak in between.
    scored = []
    for tok, df in df_counts.items():
        idf = np.log(n_docs / df) if df < n_docs else 0.0
        scored.append((tok, df * idf, df))
    scored.sort(key=lambda t: t[1], reverse=True)

    return {
        "tokens": [{"t": tok, "df": df} for tok, _, df in scored[:top_k]],
        "n_unique_tokens": len(df_counts),
    }


def infer_dtype(values: list[Any]) -> str:
    """Infer whether a column is numeric, categorical, or text."""
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
    if not str_vals:
        return "categorical"
    avg_len = np.mean([len(s) for s in str_vals])
    if avg_len > 50:
        return "text"
    # v1.1: multi-token, high-cardinality short strings (product names, descriptions)
    # behave like text, not categories — token decomposition matters for these.
    avg_tokens = np.mean([len(s.split()) for s in str_vals])
    uniqueness = len(set(str_vals)) / len(str_vals)
    if avg_tokens >= 2 and uniqueness > 0.5:
        return "text"
    return "categorical"


def build_sketch(
    values: list[Any],
    dtype: str,
    name: str = "",
    is_time_indexed: bool = False,
    acf_lags: list[int] | None = None,
    top_k: int = 10,
    budget: int = 512,
    text_sketch_mode: str = "both",
    text_token_top_k: int = 30,
) -> str:
    """
    Build a sketch string for a single column. Deterministically truncated to budget.

    For text columns, text_sketch_mode controls content (v1.1):
      - "statistical": aggregate stats only (maximum privacy; pre-v1.1 behaviour)
      - "tokens":      top TF-IDF tokens only
      - "both":        stats + tokens (default; best signal detection)
    """
    if dtype == "numeric":
        arr = np.array(
            [float(v) if v is not None else np.nan for v in values],
            dtype=np.float64,
        )
        sketch_dict = _numeric_sketch(arr, is_time_indexed, acf_lags)
    elif dtype == "text" and text_sketch_mode in ("tokens", "both"):
        if text_sketch_mode == "both":
            sketch_dict = _categorical_sketch(values, top_k)
            sketch_dict["type"] = "text"
        else:
            sketch_dict = {"type": "text", "n": len([v for v in values if v is not None])}
        tok = _token_sketch(values, text_token_top_k)
        sketch_dict["top_tokens"] = tok["tokens"]
        if "n_unique_tokens" in tok:
            sketch_dict["n_unique_tokens"] = tok["n_unique_tokens"]
        # Token sketches need more room; expand budget proportionally
        budget = max(budget, 1024)
    else:
        sketch_dict = _categorical_sketch(values, top_k)
        if dtype == "text":
            sketch_dict["type"] = "text"

    if name:
        sketch_dict["column_name"] = name

    raw = json.dumps(sketch_dict, separators=(",", ":"), sort_keys=True)
    if len(raw) > budget:
        raw = raw[: budget - 3] + "..."
    return raw


def to_numeric_array(values: list[Any]) -> np.ndarray:
    """Convert values list to float64 numpy array (NaN for missing)."""
    result = []
    for v in values:
        if v is None:
            result.append(np.nan)
        else:
            try:
                result.append(float(v))
            except (ValueError, TypeError):
                result.append(np.nan)
    return np.array(result, dtype=np.float64)

"""
Main screening pipeline orchestrator (Algorithm 1 from the paper).

Ties together sketching, embedding, cosine significance, dependence tests,
lag scan, p-value combination, and BH-FDR.
"""
import logging
import numpy as np
from typing import Any, Optional

from config import PipelineConfig
from core.sketcher import build_sketch, infer_dtype, to_numeric_array
from core.embedder import Embedder
from core.cosine_test import cosine_monte_carlo
from core.dependence import hsic_permutation_test, dcor_permutation_test, lag_scan
from core.combiner import (
    fisher_combine, brown_combine, alignment_score, benjamini_hochberg,
)

logger = logging.getLogger(__name__)


def _auto_block_size(is_time_indexed: bool, n: int) -> int:
    if not is_time_indexed:
        return 1
    return min(7, max(2, n // 20))


def _auto_lag_window(n: int) -> int:
    return min(14, max(1, n // 10))


def _detect_ar1_needed(values: np.ndarray, threshold: float = 0.4) -> bool:
    clean = values[~np.isnan(values)]
    if len(clean) < 12:
        return False
    centered = clean - np.mean(clean)
    var = np.dot(centered, centered)
    if var == 0:
        return False
    acf1 = np.dot(centered[:-1], centered[1:]) / var
    return abs(acf1) > threshold


def _evaluate_structure(
    lag_result: Optional[dict],
    ccf_threshold: float,
    peak_ratio: float,
) -> tuple[bool, Optional[dict]]:
    """
    Structural signal detection (v1.1).

    A feature has temporal structure when the lag scan finds a clear peak:
    |CCF| at the best lag exceeds ccf_threshold AND stands out from the
    off-peak background by at least peak_ratio. This is reported
    independently of statistical significance: autocorrelation-aware nulls
    correctly raise the significance bar for seasonal features, so genuine
    structure can exist below the q-value threshold. Both labels matter.
    """
    if not lag_result or not lag_result.get("ccf_by_lag"):
        return False, None

    best_lag = lag_result["best_lag"]
    best_ccf = abs(lag_result["best_ccf"])
    off_peak = [abs(v) for k, v in lag_result["ccf_by_lag"].items() if k != best_lag]

    if best_ccf < ccf_threshold or not off_peak:
        return False, None

    mean_off_peak = float(np.mean(off_peak))
    dominance = best_ccf / max(mean_off_peak, 1e-9)
    has_structure = dominance >= peak_ratio

    detail = {
        "best_lag": best_lag,
        "best_ccf": round(lag_result["best_ccf"], 6),
        "peak_dominance": round(dominance, 3),
    }
    return has_structure, detail if has_structure else None


def _synthesize_fourier_features(n_rows: int, periods: list[float]) -> list[dict]:
    """
    Fourier feature expansion (v1.1, explicit opt-in via config.fourier_periods).

    Synthesises sin/cos pairs at each requested period, indexed by row
    position (assumes regular sampling). These are screened alongside user
    features and clearly tagged as derived, so seasonal alignment between
    the target and a known period can attain proper statistical significance
    rather than relying on structure detection alone.
    """
    t = np.arange(n_rows, dtype=np.float64)
    features = []
    for p in periods:
        if p <= 1:
            continue
        w = 2.0 * np.pi / p
        p_label = f"{p:g}"
        features.append({
            "name": f"fourier_sin_p{p_label}",
            "values": np.sin(w * t).tolist(),
            "dtype": "numeric",
            "is_time_indexed": True,
            "derived": True,
            "derived_from": f"sin(2*pi*t/{p_label}), t=row index",
        })
        features.append({
            "name": f"fourier_cos_p{p_label}",
            "values": np.cos(w * t).tolist(),
            "dtype": "numeric",
            "is_time_indexed": True,
            "derived": True,
            "derived_from": f"cos(2*pi*t/{p_label}), t=row index",
        })
    return features


def run_screening_pipeline(
    y_values: list[Any],
    y_name: str,
    y_dtype: str,
    x_columns: list[dict],
    config: PipelineConfig,
    embedder: Optional[Embedder] = None,
    y_is_time_indexed: bool = False,
) -> dict:
    """
    Run the full screening pipeline (Algorithm 1).

    Args:
        y_values: Target column values.
        y_name: Target column name.
        y_dtype: Target dtype (numeric/categorical/text).
        x_columns: Feature dicts with keys: name, values, dtype, is_time_indexed.
        config: Pipeline configuration.
        embedder: Optional pre-loaded Embedder instance.
        y_is_time_indexed: Whether the target is time-indexed.

    Returns:
        Dict with results, warnings, and metadata.
    """
    warnings: list[str] = []
    rng_seed = config.random_seed

    # ── Step 0: Fourier feature expansion (v1.1, opt-in) ─────────────────
    if config.fourier_periods:
        n_rows = len(y_values)
        fourier_cols = _synthesize_fourier_features(n_rows, config.fourier_periods)
        if fourier_cols:
            x_columns = list(x_columns) + fourier_cols
            logger.info(
                f"Fourier expansion: added {len(fourier_cols)} derived features "
                f"at periods {config.fourier_periods}"
            )

    # ── Step 1: Initialize embedder ──────────────────────────────────────
    if embedder is None:
        embedder = Embedder(
            tier1_model=config.tier1_model,
            tier2_model=config.tier2_model,
            use_onnx=config.use_onnx,
        )

    # ── Step 2: Build Y sketch and embed ─────────────────────────────────
    logger.info("Building Y sketch and embedding")
    y_sketch = build_sketch(
        y_values, y_dtype, name=y_name,
        is_time_indexed=y_is_time_indexed,
        acf_lags=config.acf_lags if y_is_time_indexed else None,
        top_k=config.sketch_top_k,
        budget=config.sketch_budget,
        text_sketch_mode=config.text_sketch_mode,
        text_token_top_k=config.text_token_top_k,
    )
    y_emb_t1 = embedder.embed_tier1([y_sketch])[0]

    # ── Step 3: Build all sketches and batch embed ───────────────────────
    all_sketches: list[str] = []
    for col in x_columns:
        sketch = build_sketch(
            col["values"], col["dtype"], name=col["name"],
            is_time_indexed=col.get("is_time_indexed", False),
            acf_lags=config.acf_lags if col.get("is_time_indexed") else None,
            top_k=config.sketch_top_k,
            budget=config.sketch_budget,
            text_sketch_mode=config.text_sketch_mode,
            text_token_top_k=config.text_token_top_k,
        )
        all_sketches.append(sketch)

    logger.info(f"Batch embedding {len(all_sketches)} sketches (tier 1)")
    x_embs_t1 = embedder.embed_tier1(all_sketches)

    # ── Step 4: Per-feature testing ──────────────────────────────────────
    results: list[dict] = []

    for idx, col in enumerate(x_columns):
        col_name = col["name"]
        col_values = col["values"]
        col_dtype = col["dtype"]
        col_time = col.get("is_time_indexed", False)
        sketch = all_sketches[idx]
        x_emb = x_embs_t1[idx]

        cosine_obs = embedder.cosine_similarity(x_emb, y_emb_t1)

        block_size = config.block_size or _auto_block_size(col_time, len(col_values))
        lag_window = config.lag_window or _auto_lag_window(len(col_values))

        use_ar1 = config.enable_ar1_null
        if col_dtype == "numeric" and col_time and not config.enable_ar1_null:
            arr = to_numeric_array(col_values)
            if _detect_ar1_needed(arr, config.ar1_acf_threshold):
                use_ar1 = True
                logger.info(f"  Auto-enabled AR(1) null for {col_name}")

        # Cosine Monte Carlo
        cos_result = cosine_monte_carlo(
            observed_cosine=cosine_obs,
            y_embedding=y_emb_t1,
            values=col_values,
            dtype=col_dtype,
            column_name=col_name,
            embed_fn=embedder.embed_tier1,
            B=config.cosine_null_draws,
            is_time_indexed=col_time,
            block_size=block_size,
            use_ar1=use_ar1,
            replacement_fraction=config.replacement_fraction,
            token_shuffle_prob=config.token_shuffle_prob,
            acf_lags=config.acf_lags if col_time else None,
            top_k=config.sketch_top_k,
            budget=config.sketch_budget,
            seed=rng_seed + idx,
            text_sketch_mode=config.text_sketch_mode,
            text_token_top_k=config.text_token_top_k,
        )

        # Dependence tests (numeric only, aligned lengths)
        hsic_result = None
        dcor_result = None
        lag_result = None

        if col_dtype == "numeric":
            x_arr = to_numeric_array(col_values)
            y_arr = to_numeric_array(y_values)

            if len(x_arr) == len(y_arr):
                hsic_result = hsic_permutation_test(
                    x_arr, y_arr,
                    n_perm=config.dependence_permutations,
                    block_size=block_size if col_time else None,
                    seed=rng_seed + idx + 10000,
                )
                dcor_result = dcor_permutation_test(
                    x_arr, y_arr,
                    n_perm=config.dependence_permutations,
                    block_size=block_size if col_time else None,
                    seed=rng_seed + idx + 20000,
                )
                if col_time:
                    lag_result = lag_scan(
                        x_arr, y_arr,
                        max_lag=lag_window,
                        positive_only=config.restrict_positive_lags,
                    )
            else:
                warnings.append(
                    f"Feature '{col_name}' length ({len(x_arr)}) != "
                    f"Y length ({len(y_arr)}); skipping HSIC/dCor/lag"
                )

        # Collect raw p-values
        raw_pvals = [cos_result["p_value"]]
        if hsic_result and not np.isnan(hsic_result.get("p_value", np.nan)):
            raw_pvals.append(hsic_result["p_value"])
        if dcor_result and not np.isnan(dcor_result.get("p_value", np.nan)):
            raw_pvals.append(dcor_result["p_value"])

        combined_p = fisher_combine(raw_pvals) if len(raw_pvals) > 1 else raw_pvals[0]
        align_score = alignment_score(raw_pvals)

        dcor_stat = None
        if dcor_result and not np.isnan(dcor_result.get("statistic", np.nan)):
            dcor_stat = round(dcor_result["statistic"], 6)

        # v1.1: structural signal detection, independent of significance
        has_structure, structure_detail = _evaluate_structure(
            lag_result,
            config.structure_ccf_threshold,
            config.structure_peak_ratio,
        )

        results.append({
            "name": col_name,
            "inferred_dtype": col_dtype,
            "sketch_preview": sketch[:200],
            "cosine_score": round(cosine_obs, 6),
            "cosine_p_value": cos_result["p_value"],
            "cosine_null_mean": round(cos_result["null_mean"], 6),
            "cosine_null_std": round(cos_result["null_std"], 6),
            "hsic_p_value": hsic_result["p_value"] if hsic_result else None,
            "dcor_p_value": dcor_result["p_value"] if dcor_result else None,
            "dcor_statistic": dcor_stat,
            "best_lag": lag_result["best_lag"] if lag_result else None,
            "best_lag_ccf": round(lag_result["best_ccf"], 6) if lag_result else None,
            "has_structure": has_structure,
            "structure_detail": structure_detail,
            "derived": col.get("derived", False),
            "derived_from": col.get("derived_from"),
            "combined_p_value": combined_p,
            "alignment_score": round(align_score, 4),
            "_raw_pvals": raw_pvals,
        })

    # ── Step 5: BH-FDR across features ───────────────────────────────────
    combined_ps = [r["combined_p_value"] for r in results]
    q_values = benjamini_hochberg(combined_ps, config.fdr_alpha)

    for i, r in enumerate(results):
        qv = q_values[i]
        r["q_value"] = round(qv, 6) if not np.isnan(qv) else None
        r["significant"] = bool(qv < config.fdr_alpha) if not np.isnan(qv) else False

    # ── Step 6: Tier-2 confirmation (optional) ───────────────────────────
    if config.tier2_top_k and config.tier2_top_k > 0:
        try:
            logger.info(f"Running tier-2 confirmation on top {config.tier2_top_k}")
            sorted_by_score = sorted(
                results, key=lambda r: r["alignment_score"], reverse=True
            )
            top_k_names = {r["name"] for r in sorted_by_score[: config.tier2_top_k]}

            y_emb_t2 = embedder.embed_tier2([y_sketch])[0]

            top_k_indices = [
                i for i, r in enumerate(results) if r["name"] in top_k_names
            ]
            top_k_sketches = [all_sketches[i] for i in top_k_indices]

            if top_k_sketches:
                t2_embs = embedder.embed_tier2(top_k_sketches)
                for local_idx, global_idx in enumerate(top_k_indices):
                    t2_cosine = embedder.cosine_similarity(t2_embs[local_idx], y_emb_t2)
                    t2_B = min(config.cosine_null_draws, 100)
                    col = x_columns[global_idx]
                    col_time = col.get("is_time_indexed", False)
                    col_block_size = config.block_size or _auto_block_size(col_time, len(col["values"]))
                    t2_result = cosine_monte_carlo(
                        observed_cosine=t2_cosine,
                        y_embedding=y_emb_t2,
                        values=col["values"],
                        dtype=col["dtype"],
                        column_name=col["name"],
                        embed_fn=embedder.embed_tier2,
                        B=t2_B,
                        is_time_indexed=col_time,
                        block_size=col_block_size,
                        use_ar1=False,
                        replacement_fraction=config.replacement_fraction,
                        token_shuffle_prob=config.token_shuffle_prob,
                        top_k=config.sketch_top_k,
                        budget=config.sketch_budget,
                        seed=rng_seed + global_idx + 50000,
                    )
                    results[global_idx]["tier2_cosine_score"] = round(t2_cosine, 6)
                    results[global_idx]["tier2_cosine_p_value"] = t2_result["p_value"]
        except Exception as e:
            logger.warning(f"Tier-2 confirmation failed (skipped): {e}")
            warnings.append(
                f"Tier-2 confirmation skipped — could not load model "
                f"'{config.tier2_model}'. Place the model locally or disable "
                f"tier2_top_k. Error: {e}"
            )

    # Sort by alignment score descending
    results.sort(key=lambda r: r["alignment_score"], reverse=True)

    # Remove internal fields
    for r in results:
        r.pop("_raw_pvals", None)

    return {
        "results": results,
        "warnings": warnings,
        "n_features_screened": len(results),
        "n_significant": sum(1 for r in results if r.get("significant", False)),
        "n_with_structure": sum(1 for r in results if r.get("has_structure", False)),
        "fdr_alpha": config.fdr_alpha,
        "tier1_model": config.tier1_model,
        "tier2_model": config.tier2_model if config.tier2_top_k else None,
    }

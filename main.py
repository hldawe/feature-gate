"""
Feature Gate API -- a governance-friendly, training-free screening gate.

Endpoints:
    POST /screen   -- Run the full screening pipeline
    GET  /health   -- Health check and model status
    GET  /config   -- Return default configuration
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import PipelineConfig
from schemas import ScreeningRequest, ScreeningResponse, FeatureResult
from core.sketcher import infer_dtype
from core.embedder import Embedder
from core.pipeline import run_screening_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("feature_gate")

# ── Global state ─────────────────────────────────────────────────────────────

_embedder: Optional[Embedder] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load tier-1 model on startup for fast first request."""
    global _embedder
    _defaults = PipelineConfig()
    logger.info(f"Pre-loading tier-1 embedding model: {_defaults.tier1_model}")
    _embedder = Embedder(
        tier1_model=_defaults.tier1_model,
        tier2_model=_defaults.tier2_model,
        use_onnx=_defaults.use_onnx,
    )
    _ = _embedder.tier1  # trigger lazy load
    logger.info("Tier-1 model ready")
    yield
    logger.info("Shutting down")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Feature Gate",
    description=(
        "A model-free, governance-friendly screening pipeline that determines "
        "whether features in X could plausibly inform a target Y, using frozen "
        "sentence-transformer embeddings over privacy-preserving column sketches, "
        "calibrated Monte Carlo significance tests, and model-free dependence measures."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "tier1_loaded": _embedder is not None and _embedder._tier1 is not None,
        "tier2_loaded": _embedder is not None and _embedder._tier2 is not None,
    }


@app.get("/config")
async def default_config():
    """Return the default pipeline configuration."""
    return PipelineConfig().model_dump()


@app.post("/screen", response_model=ScreeningResponse)
async def screen(request: ScreeningRequest):
    """
    Run the full screening pipeline.

    Accepts a target column Y and one or more feature columns X.
    Returns a ranked shortlist with calibrated p/q-values, suggested lags,
    and auditable artifacts.
    """
    t0 = time.time()
    config = request.config

    # Validate input lengths
    y_len = len(request.y.values)
    warnings = []
    if y_len < 10:
        warnings.append(
            f"Target column has only {y_len} values; results may be unreliable"
        )

    # Check for model override mismatch
    if _embedder is not None:
        if config.tier1_model != _embedder.tier1_model_name:
            warnings.append(
                f"Requested tier1_model '{config.tier1_model}' differs from "
                f"pre-loaded '{_embedder.tier1_model_name}'; using pre-loaded model. "
                f"Restart the server with updated config to change models."
            )
        if config.tier2_model != _embedder.tier2_model_name:
            warnings.append(
                f"Requested tier2_model '{config.tier2_model}' differs from "
                f"pre-loaded '{_embedder.tier2_model_name}'; using pre-loaded model. "
                f"Restart the server with updated config to change models."
            )

    # Infer dtypes where not provided
    y_dtype = request.y.dtype or infer_dtype(request.y.values)

    x_columns = []
    for col in request.x_columns:
        col_dtype = col.dtype or infer_dtype(col.values)
        x_columns.append({
            "name": col.name,
            "values": col.values,
            "dtype": col_dtype,
            "is_time_indexed": col.is_time_indexed,
        })

    # Run pipeline
    try:
        pipeline_result = run_screening_pipeline(
            y_values=request.y.values,
            y_name=request.y.name,
            y_dtype=y_dtype,
            x_columns=x_columns,
            config=config,
            embedder=_embedder,
            y_is_time_indexed=request.y.is_time_indexed,
        )
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    elapsed = time.time() - t0
    logger.info(
        f"Screening complete: {pipeline_result['n_features_screened']} features, "
        f"{pipeline_result['n_significant']} significant, {elapsed:.1f}s"
    )

    # Map to response schema
    feature_results = []
    for r in pipeline_result["results"]:
        feature_results.append(FeatureResult(
            name=r["name"],
            inferred_dtype=r["inferred_dtype"],
            sketch_preview=r["sketch_preview"],
            cosine_score=r["cosine_score"],
            cosine_p_value=r["cosine_p_value"],
            cosine_null_mean=r["cosine_null_mean"],
            cosine_null_std=r["cosine_null_std"],
            hsic_p_value=r.get("hsic_p_value"),
            dcor_p_value=r.get("dcor_p_value"),
            dcor_statistic=r.get("dcor_statistic"),
            best_lag=r.get("best_lag"),
            best_lag_ccf=r.get("best_lag_ccf"),
            has_structure=r.get("has_structure", False),
            structure_detail=r.get("structure_detail"),
            derived=r.get("derived", False),
            derived_from=r.get("derived_from"),
            combined_p_value=r["combined_p_value"],
            q_value=r["q_value"] if r.get("q_value") is not None else 1.0,
            alignment_score=r["alignment_score"],
            significant=r.get("significant", False),
            tier2_cosine_score=r.get("tier2_cosine_score"),
            tier2_cosine_p_value=r.get("tier2_cosine_p_value"),
        ))

    all_warnings = warnings + pipeline_result.get("warnings", [])

    return ScreeningResponse(
        n_features_screened=pipeline_result["n_features_screened"],
        n_significant=pipeline_result["n_significant"],
        n_with_structure=pipeline_result.get("n_with_structure", 0),
        fdr_alpha=pipeline_result["fdr_alpha"],
        tier1_model=pipeline_result["tier1_model"],
        tier2_model=pipeline_result.get("tier2_model"),
        results=feature_results,
        warnings=all_warnings,
    )


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

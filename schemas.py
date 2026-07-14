"""
API request and response schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, Any, Literal
from config import PipelineConfig


# ── Request ──────────────────────────────────────────────────────────────────

class ColumnPayload(BaseModel):
    name: str = Field(..., description="Column name")
    values: list[Any] = Field(..., description="Column values (numeric, string, or mixed)")
    dtype: Optional[Literal["numeric", "categorical", "text"]] = Field(
        None, description="Force dtype: 'numeric', 'categorical', 'text'. Auto-detected if None."
    )
    is_time_indexed: bool = Field(False, description="Whether values are ordered by time")


class ScreeningRequest(BaseModel):
    """POST body for /screen endpoint."""
    y: ColumnPayload = Field(..., description="Target column")
    x_columns: list[ColumnPayload] = Field(..., min_length=1, description="Feature columns to screen")
    config: PipelineConfig = Field(default_factory=PipelineConfig, description="Pipeline parameters")


# ── Response ─────────────────────────────────────────────────────────────────

class FeatureResult(BaseModel):
    name: str
    inferred_dtype: str
    sketch_preview: str = Field(description="Truncated sketch string")

    # Cosine
    cosine_score: float
    cosine_p_value: float
    cosine_null_mean: float
    cosine_null_std: float

    # Dependence tests (numeric only)
    hsic_p_value: Optional[float] = None
    dcor_p_value: Optional[float] = None
    dcor_statistic: Optional[float] = None

    # Lag scan (numeric + time-indexed only)
    best_lag: Optional[int] = None
    best_lag_ccf: Optional[float] = None

    # Structural signal (v1.1) — temporal structure independent of significance
    has_structure: bool = Field(
        False,
        description="Lag scan found a clear periodic/lagged peak, reported independently "
                    "of statistical significance. Structure without significance warrants "
                    "human review, e.g. via fourier_periods expansion."
    )
    structure_detail: Optional[dict] = Field(
        None, description="best_lag, best_ccf, peak_dominance when has_structure is true"
    )

    # Derived feature provenance (v1.1)
    derived: bool = Field(False, description="True for features synthesised by Feature Gate (e.g. Fourier)")
    derived_from: Optional[str] = Field(None, description="Generation formula for derived features")

    # Combined
    combined_p_value: float
    q_value: float = Field(description="BH-FDR adjusted p-value")
    alignment_score: float = Field(description="Ranking heuristic: -sum(log10(p + eps))")
    significant: bool = Field(description="q_value < fdr_alpha")

    # Tier-2 (optional)
    tier2_cosine_score: Optional[float] = None
    tier2_cosine_p_value: Optional[float] = None


class ScreeningResponse(BaseModel):
    """Response from /screen endpoint."""
    n_features_screened: int
    n_significant: int
    n_with_structure: int = Field(0, description="Features with detected temporal structure (v1.1)")
    fdr_alpha: float
    tier1_model: str
    tier2_model: Optional[str] = None
    results: list[FeatureResult] = Field(description="Ranked by alignment_score descending")
    warnings: list[str] = Field(default_factory=list)

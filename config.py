"""
Default parameters from the paper's recommended practical settings.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class PipelineConfig(BaseModel):
    # --- Cosine Monte Carlo ---
    cosine_null_draws: int = Field(200, ge=50, le=2000, description="B: null draws per feature")

    # --- Text/categorical null perturbation ---
    replacement_fraction: float = Field(0.20, ge=0.0, le=1.0, description="r: fraction of entries replaced in text null")
    token_shuffle_prob: float = Field(0.30, ge=0.0, le=1.0, description="beta: token-shuffle probability for sketch string")

    # --- Time-series ---
    block_size: Optional[int] = Field(None, ge=1, description="Block size for block permutation (auto-detected if None)")
    enable_ar1_null: bool = Field(False, description="Enable AR(1) null resampling for numeric time series")
    ar1_acf_threshold: float = Field(0.4, ge=0.0, le=1.0, description="ACF(1) threshold to auto-enable AR(1) null")

    # --- Lag scan ---
    lag_window: Optional[int] = Field(None, ge=0, description="L: lag scan window (auto-detected if None)")
    restrict_positive_lags: bool = Field(False, description="Restrict to tau >= 0 (prevents leakage)")

    # --- HSIC / dCor ---
    dependence_permutations: int = Field(200, ge=50, le=2000, description="Permutations for HSIC/dCor p-values")

    # --- Multiple testing ---
    fdr_alpha: float = Field(0.05, ge=0.001, le=0.5, description="BH-FDR significance level")

    # --- Embedding tiers ---
    tier1_model: str = Field("models/all-MiniLM-L6-v2", description="Fast screening model (HuggingFace name or local path)")
    tier2_model: str = Field("BAAI/bge-base-en-v1.5", description="Confirmation model for top-k")
    use_onnx: bool = Field(True, description="Use ONNX backend for speed")
    tier2_top_k: Optional[int] = Field(None, description="Re-embed top-k features with tier-2 (None = skip)")

    # --- Sketch ---
    sketch_top_k: int = Field(10, ge=1, le=50, description="Top-k values in categorical sketch")
    sketch_budget: int = Field(512, ge=128, le=2048, description="Max character budget for sketch string")
    acf_lags: list[int] = Field(default=[1, 7, 30], description="ACF fingerprint lags for numeric time series")

    # --- Text token sketching (v1.1) ---
    text_sketch_mode: Literal["statistical", "tokens", "both"] = Field(
        "both",
        description="Text column sketching: 'statistical' (aggregate stats only, maximum privacy), "
                    "'tokens' (top TF-IDF tokens only), 'both' (default; stats + tokens)."
    )
    text_token_top_k: int = Field(30, ge=5, le=100, description="Top-k tokens included in text token sketch")

    # --- Structural signal detection (v1.1) ---
    structure_ccf_threshold: float = Field(
        0.30, ge=0.0, le=1.0,
        description="Min |CCF| at best lag to flag temporal structure independently of significance"
    )
    structure_peak_ratio: float = Field(
        1.5, ge=1.0, le=10.0,
        description="Best-lag |CCF| must exceed this multiple of the mean off-peak |CCF|"
    )

    # --- Fourier feature expansion (v1.1) ---
    fourier_periods: Optional[list[float]] = Field(
        None,
        description="If set (e.g. [7, 30.44, 365.25]), synthesise sin/cos features at these periods "
                    "(in row units) and screen them alongside user features. Assumes regular sampling."
    )

    # --- Misc ---
    random_seed: int = Field(42, description="Seed for reproducibility")

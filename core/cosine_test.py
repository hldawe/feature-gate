"""
Cosine significance with type-aware Monte Carlo nulls (Section 3.3).

Generates null columns, re-sketches, re-embeds, and computes null cosine distribution.
"""
import numpy as np
from typing import Any

from core.sketcher import build_sketch


def _permute_numeric(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simple permutation null for numeric column."""
    perm = values.copy()
    rng.shuffle(perm)
    return perm


def _block_permute_numeric(
    values: np.ndarray, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Block permutation: shuffle blocks of contiguous values."""
    n = len(values)
    n_blocks = max(1, n // block_size)
    padded_len = n_blocks * block_size
    padded = values[:padded_len].copy()
    blocks = padded.reshape(n_blocks, block_size)
    rng.shuffle(blocks)
    result = blocks.flatten()
    if padded_len < n:
        result = np.concatenate([result, values[padded_len:]])
    return result


def _ar1_resample(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """AR(1) resampling: preserve autocorrelation structure, break alignment."""
    clean = values[~np.isnan(values)]
    if len(clean) < 3:
        return _permute_numeric(values, rng)

    mu = np.mean(clean)
    centered = clean - mu
    if np.var(centered) == 0:
        return _permute_numeric(values, rng)
    phi = np.corrcoef(centered[:-1], centered[1:])[0, 1]
    phi = np.clip(phi, -0.99, 0.99)
    residual_std = np.std(centered) * np.sqrt(1 - phi ** 2)

    result = np.empty(len(clean))
    result[0] = rng.normal(0, np.std(centered))
    for i in range(1, len(clean)):
        result[i] = phi * result[i - 1] + rng.normal(0, residual_std)
    result += mu

    out = np.full_like(values, np.nan)
    mask = ~np.isnan(values)
    out[mask] = result[: mask.sum()]
    return out


def _parametric_numeric_null(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Parametric N(mu, sigma^2) resampling to destroy fine structure."""
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return values.copy()
    mu, sigma = np.mean(clean), np.std(clean)
    out = np.full_like(values, np.nan)
    mask = ~np.isnan(values)
    out[mask] = rng.normal(mu, max(sigma, 1e-8), size=mask.sum())
    return out


def generate_numeric_null(
    values: np.ndarray,
    rng: np.random.Generator,
    is_time_indexed: bool = False,
    block_size: int = 7,
    use_ar1: bool = False,
) -> np.ndarray:
    """Generate a null numeric column using mixture of strategies."""
    strategy = rng.choice(["permute", "parametric"])
    if is_time_indexed:
        if use_ar1:
            strategy = rng.choice(["block_permute", "ar1"])
        else:
            strategy = rng.choice(["block_permute", "parametric"])

    if strategy == "permute":
        return _permute_numeric(values, rng)
    elif strategy == "block_permute":
        return _block_permute_numeric(values, block_size, rng)
    elif strategy == "ar1":
        return _ar1_resample(values, rng)
    else:
        return _parametric_numeric_null(values, rng)


def generate_categorical_null(
    values: list[Any],
    rng: np.random.Generator,
    replacement_fraction: float = 0.20,
) -> list[Any]:
    """Two-stage text/categorical null: shuffle + partial replacement."""
    shuffled = list(values)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_replace = max(1, int(n * replacement_fraction))
    replace_idx = rng.choice(n, size=n_replace, replace=False)
    donor_idx = rng.choice(n, size=n_replace, replace=True)
    for i, d in zip(replace_idx, donor_idx):
        shuffled[int(i)] = shuffled[int(d)]

    return shuffled


def generate_text_null(
    values: list[Any],
    rng: np.random.Generator,
    replacement_fraction: float = 0.20,
) -> list[Any]:
    """
    Token-level text null (v1.1).

    Row shuffling alone leaves token document frequencies — and therefore the
    token sketch — unchanged, which would make the null degenerate. Instead we
    rebuild a fraction of rows by sampling words from the column's global word
    pool, preserving per-row word counts and approximate marginal frequencies
    while scrambling document frequency and co-occurrence structure. This
    perturbs the top-k token sketch within the column's own vocabulary.
    """
    str_vals = [str(v) if v is not None else "" for v in values]
    word_pool: list[str] = []
    for s in str_vals:
        word_pool.extend(s.split())
    if not word_pool:
        return generate_categorical_null(values, rng, replacement_fraction)

    frac = max(replacement_fraction, 0.30)  # text needs stronger perturbation
    n = len(str_vals)
    n_replace = max(1, int(n * frac))
    replace_idx = set(int(i) for i in rng.choice(n, size=n_replace, replace=False))

    pool_arr = np.array(word_pool, dtype=object)
    out: list[Any] = []
    for i, s in enumerate(str_vals):
        if i in replace_idx and s:
            k = max(1, len(s.split()))
            sampled = pool_arr[rng.integers(0, len(pool_arr), size=k)]
            out.append(" ".join(sampled))
        else:
            out.append(values[i])

    # Shuffle row order too (harmless for sketch, breaks any residual alignment)
    order = rng.permutation(n)
    return [out[int(i)] for i in order]


def _token_shuffle_sketch(sketch_str: str, rng: np.random.Generator, prob: float = 0.30) -> str:
    """Randomly shuffle tokens in the sketch string with probability beta."""
    tokens = sketch_str.split(",")
    if len(tokens) <= 1:
        return sketch_str
    n = len(tokens)
    n_swap = max(1, int(n * prob))
    for _ in range(n_swap):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return ",".join(tokens)


def cosine_monte_carlo(
    observed_cosine: float,
    y_embedding: np.ndarray,
    values: list[Any],
    dtype: str,
    column_name: str,
    embed_fn: callable,
    B: int = 200,
    is_time_indexed: bool = False,
    block_size: int = 7,
    use_ar1: bool = False,
    replacement_fraction: float = 0.20,
    token_shuffle_prob: float = 0.30,
    acf_lags: list[int] | None = None,
    top_k: int = 10,
    budget: int = 512,
    seed: int = 42,
    embed_batch_size: int = 64,
    text_sketch_mode: str = "both",
    text_token_top_k: int = 30,
) -> dict:
    """
    Monte Carlo p-value for cosine significance (Section 3.3).

    Returns dict with p_value, null_mean, null_std, null_cosines.
    """
    rng = np.random.default_rng(seed)

    if dtype == "numeric":
        arr = np.array(
            [float(v) if v is not None else np.nan for v in values],
            dtype=np.float64,
        )

    # Phase 1: generate all null sketches
    null_sketches: list[str] = []
    for b in range(B):
        if dtype == "numeric":
            null_vals_arr = generate_numeric_null(
                arr, rng, is_time_indexed, block_size, use_ar1
            )
            null_vals = null_vals_arr.tolist()
        elif dtype == "text":
            null_vals = generate_text_null(values, rng, replacement_fraction)
        else:
            null_vals = generate_categorical_null(values, rng, replacement_fraction)

        null_sketch = build_sketch(
            null_vals, dtype, name=column_name,
            is_time_indexed=is_time_indexed,
            acf_lags=acf_lags, top_k=top_k, budget=budget,
            text_sketch_mode=text_sketch_mode,
            text_token_top_k=text_token_top_k,
        )

        if token_shuffle_prob > 0:
            null_sketch = _token_shuffle_sketch(null_sketch, rng, token_shuffle_prob)

        null_sketches.append(null_sketch)

    # Phase 2: batch embed all null sketches
    null_embs = embed_fn(null_sketches)

    # Phase 3: compute all cosines
    null_cosines = np.array(
        [float(np.dot(emb, y_embedding)) for emb in null_embs],
        dtype=np.float64,
    )

    # Monte Carlo p-value: (1 + #{null >= obs}) / (B + 1)
    p_value = (1 + np.sum(null_cosines >= observed_cosine)) / (B + 1)

    return {
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_cosines)),
        "null_std": float(np.std(null_cosines)),
        "null_cosines": null_cosines,
    }

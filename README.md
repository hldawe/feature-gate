# Feature Gate

**A model-free, governance-friendly API for screening whether features in X could plausibly inform a target Y — without training any supervised model.**

Feature Gate answers the triage question every ML team faces before modelling: *"Is there enough signal in these features to justify building a model?"* It handles numeric, categorical, and free-text columns in a single pipeline, runs entirely locally on CPU, produces calibrated p-values, and outputs auditable artifacts suitable for governance review.

---

## Table of Contents

- [Why Feature Gate](#why-feature-gate)
- [How It Works](#how-it-works)
  - [Pipeline Overview](#pipeline-overview)
  - [Step 1: Sketch Construction](#step-1-sketch-construction)
  - [Step 2: Frozen Embedding](#step-2-frozen-embedding)
  - [Step 3: Cosine Significance Testing](#step-3-cosine-significance-testing)
  - [Step 4: Model-Free Dependence Tests](#step-4-model-free-dependence-tests)
  - [Step 5: Lag Scan](#step-5-lag-scan)
  - [Step 6: P-Value Combination and Multiple Testing](#step-6-p-value-combination-and-multiple-testing)
  - [Two-Tier Embedding Architecture](#two-tier-embedding-architecture)
- [Quick Start](#quick-start)
  - [Local Installation](#local-installation)
  - [Docker](#docker)
  - [Running the Example](#running-the-example)
- [Demo Datasets (Real-World Data)](#demo-datasets-real-world-data)
  - [Dataset 1: Restaurant Tips](#dataset-1-restaurant-tips)
  - [Dataset 2: Car Fuel Economy](#dataset-2-car-fuel-economy)
  - [Dataset 3: Airline Passengers (Time Series)](#dataset-3-airline-passengers-time-series)
- [API Reference](#api-reference)
  - [POST /screen](#post-screen)
  - [GET /health](#get-health)
  - [GET /config](#get-config)
- [Configuration Guide](#configuration-guide)
  - [Parameter Reference](#parameter-reference)
  - [Tuning for Your Domain](#tuning-for-your-domain)
- [Understanding the Output](#understanding-the-output)
  - [Reading the Results](#reading-the-results)
  - [Decision Framework](#decision-framework)
- [Architecture and Project Structure](#architecture-and-project-structure)
- [Performance Considerations](#performance-considerations)
- [What's New in v1.1](#whats-new-in-v11)
- [Limitations and Caveats](#limitations-and-caveats)
- [Troubleshooting](#troubleshooting)

---

## Why Feature Gate

Most applied ML work begins with an implicit decision: *is there enough signal to justify modelling Y from X?* In practice, teams either skip this step entirely (jumping straight to model fitting), run ad hoc correlations on the numeric columns they happen to notice, or rely on domain intuition. This creates several problems:

- **Wasted compute**: weeks spent training models on uninformative features.
- **Missed signal**: text and categorical columns are often ignored because they don't fit neatly into correlation analyses.
- **Temporal misalignment**: features that are predictive at a lag are invisible to contemporaneous tests.
- **Governance gaps**: no auditable record of why features were included or excluded.
- **Multiplicity blindness**: testing dozens of features without adjusting for multiple comparisons inflates false discovery rates.

Feature Gate addresses all of these by providing a single, integrated screening stage that is training-free, modality-bridging, calibrated, time-series aware, and governance-friendly.

---

## How It Works

### Pipeline Overview

```
+------------------------------------------------------------------+
|                        INPUT                                      |
|  Target Y (any type) + Feature columns X1..Xp (any types)        |
+--------------+---------------------------------------------------+
               |
               v
+------------------------------------------------------------------+
|  STEP 1: SKETCH CONSTRUCTION                                     |
|  Each column -> JSON->string summary (stats, entropy, shapes)    |
|  Privacy-preserving: no raw values leave the sketch               |
+--------------+---------------------------------------------------+
               |
               v
+------------------------------------------------------------------+
|  STEP 2: FROZEN EMBEDDING (Tier 1: all-MiniLM-L6-v2)            |
|  Sketch strings -> unit vectors via frozen sentence-transformer   |
|  Cosine similarity: cos(embed(Sj), embed(Sy))                    |
+--------------+---------------------------------------------------+
               |
               v
+------------------------------------------------------------------+
|  STEP 3: COSINE SIGNIFICANCE (Monte Carlo)                       |
|  Type-aware nulls: permutation, block-perm, AR(1), shuffle       |
|  -> p-value per feature for cosine score                          |
+--------------+---------------------------------------------------+
               |
               v
+------------------------------------------------------------------+
|  STEP 4: DEPENDENCE TESTS (numeric features only)                |
|  HSIC (RBF kernel, median heuristic) + permutation p-value       |
|  Distance correlation + permutation p-value                       |
|  Block-aware permutations for time series                         |
+--------------+---------------------------------------------------+
               |
               v
+------------------------------------------------------------------+
|  STEP 5: LAG SCAN (numeric + time-indexed only)                  |
|  Cross-correlation over tau in [-L, L]                            |
|  Reports best lag and CCF magnitude                               |
+--------------+---------------------------------------------------+
               |
               v
+------------------------------------------------------------------+
|  STEP 6: COMBINE + CORRECT                                       |
|  Fisher's method per feature -> single combined p-value           |
|  Benjamini-Hochberg FDR across all features -> q-values           |
|  Alignment score (ranking heuristic)                              |
+--------------+---------------------------------------------------+
               |
               v
+------------------------------------------------------------------+
|  OPTIONAL: TIER-2 CONFIRMATION (BAAI/bge-base-en-v1.5)          |
|  Re-embed and re-test top-k features with higher-quality model    |
+--------------+---------------------------------------------------+
               |
               v
+------------------------------------------------------------------+
|                        OUTPUT                                     |
|  Ranked shortlist with p/q-values, lags, sketches, scores         |
+------------------------------------------------------------------+
```

### Step 1: Sketch Construction

Each column is converted into a compact JSON string summarising its distributional properties. No raw data values are transmitted — only aggregate statistics. This makes the pipeline privacy-preserving by design.

**Numeric columns** produce sketches containing: min, quartiles (q25, median, q75), max, mean, median absolute deviation (MAD), skewness, kurtosis, Tukey outlier rates (fraction of values beyond 1.5x IQR), and optional autocorrelation fingerprints at configurable lags (default: 1, 7, 30) for time-indexed data.

**Categorical and text columns** produce sketches containing: top-k values with counts (default k=10), entropy of the truncated frequency histogram, number of unique values, and string-shape features (average length, digit fraction, uppercase fraction).

All sketches are deterministically truncated to a fixed character budget (default 512 characters) for reproducibility.

### Step 2: Frozen Embedding

Sketch strings are embedded into unit vectors using a frozen (non-trainable) sentence-transformer model. Because both numeric and text column sketches are represented as short text strings, they occupy the same embedding space and can be directly compared.

The cosine similarity between a feature sketch embedding and the target sketch embedding measures their "semantic alignment" — how similar their distributional profiles look in embedding space.

### Step 3: Cosine Significance Testing

A raw cosine score is not interpretable without calibration. Feature Gate generates a null distribution by:

- **Numeric columns**: Randomly choosing between permutation (or block permutation for time series) and parametric N(mu, sigma^2) resampling. Optionally, AR(1) resampling preserves autocorrelation structure while breaking alignment with Y.
- **Categorical/text columns**: A two-stage perturbation that first shuffles values to destroy row-level alignment, then replaces a fraction r (default 0.20) of entries with draws from the shuffled pool.
- **Token shuffle** (optional): Randomly swaps tokens in the sketch string itself with probability beta (default 0.30) to reduce sensitivity to sketch phrasing order.

For each null draw, the full sketch-embed-cosine pipeline is re-run. The Monte Carlo p-value is computed as: `p = (1 + count(null_cosines >= observed_cosine)) / (B + 1)`.

### Step 4: Model-Free Dependence Tests

For numeric features aligned with Y, two additional non-parametric dependence measures are computed:

**HSIC (Hilbert-Schmidt Independence Criterion)**: Uses RBF kernels with median heuristic bandwidth selection. Measures dependence in a reproducing kernel Hilbert space — captures nonlinear relationships that linear correlation misses. P-values via permutation testing (block-aware for time series).

**Distance Correlation (dCor)**: Based on pairwise Euclidean distances between observations. Has the property that dCor = 0 if and only if the variables are independent (for finite moments). Also uses permutation p-values.

### Step 5: Lag Scan

For numeric, time-indexed features, cross-correlation is computed over a window of lags tau in [-L, L]. This detects features that are predictive of Y at a temporal offset — a common pattern in forecasting (e.g., weather data predicting sales 3 days later). The best lag and its CCF magnitude are reported.

### Step 6: P-Value Combination and Multiple Testing

**Per-feature combination**: The raw p-values from cosine significance, HSIC, and dCor are combined into a single per-feature p-value using Fisher's method. For cases where the tests are strongly dependent (they share the same underlying data), Brown's correction can be applied.

**Across-feature correction**: Benjamini-Hochberg FDR is applied across all combined per-feature p-values to produce q-values. A feature is flagged as "significant" if its q-value is below the FDR threshold (default alpha = 0.05).

**Alignment score**: A non-inferential ranking heuristic computed as `-sum(log10(p + eps))` across all raw p-values. Higher scores indicate stronger evidence of a relationship. This is for ranking only, not for statistical inference.

### Two-Tier Embedding Architecture

Feature Gate uses a two-tier approach to balance speed and quality:

- **Tier 1 (`all-MiniLM-L6-v2`)**: 22.7M parameters, 384-dimensional embeddings. Extremely fast on CPU, especially with ONNX backend. Used for screening all features and generating all Monte Carlo null draws.

- **Tier 2 (`BAAI/bge-base-en-v1.5`)**: 109M parameters, 768-dimensional embeddings. Higher embedding quality with richer representation space. Used only for confirming the top-k features that survive Tier-1 screening. Activated by setting `tier2_top_k` in the config.

This design means you get the speed of MiniLM for the bulk of the computation, with the quality of BGE for the features that matter most.

---

## Quick Start

### Local Installation

```bash
# Clone or download the repo
cd feature-gate

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The server starts on `http://localhost:8000`. On first run, it will download the Tier-1 embedding model (~90MB). The auto-generated API docs are available at `http://localhost:8000/docs` (Swagger UI).

### Docker

```bash
# Build (pre-downloads the model into the image)
docker build -t feature-gate .

# Run
docker run -p 8000:8000 feature-gate
```

### Running the Example

With the server running, open a second terminal:

```bash
python example_client.py
```

This generates a synthetic dataset with 5 features (2 with real signal, 3 pure noise) and screens them against a sine-wave target. Expected behaviour:

- `correlated_signal` — should rank highest with significant q-value
- `lagged_signal_tau5` — should rank high, with best_lag near 5
- `categorical_from_y` — may show moderate cosine signal (derived from Y quantiles)
- `pure_noise` — should be non-significant
- `random_text` — should be non-significant

---

## Demo Datasets (Real-World Data)

Three real-world datasets are included in the `demo-datasets/` directory, each chosen to showcase a different Feature Gate capability. These are designed for client demonstrations — every audience member will recognise the data.

### Setup

Copy the `demo-datasets/` folder into your Feature Gate project directory, then with the server running:

```bash
python demo-datasets/run_demos.py
```

This sends all three datasets sequentially and prints a formatted results table for each. You can also send them individually:

```bash
curl -X POST http://localhost:8000/screen \
  -H "Content-Type: application/json" \
  -d @demo-datasets/demo1_tips_payload.json
```

### Dataset 1: Restaurant Tips

- **Source**: Bryant & Smith (1995), real data from a restaurant collected over 2.5 months
- **Rows**: 244 meals
- **Target**: Tip amount (dollars)
- **Features**: total_bill (numeric), party_size (numeric), sex (categorical), smoker (categorical), day_of_week (categorical), meal_time (categorical)
- **Demonstrates**: Mixed numeric + categorical screening in a single API call
- **Expected results**: `total_bill` should rank #1 as the dominant driver. `party_size` should show moderate signal. `smoker`, `day_of_week`, and `meal_time` should be filtered out as non-significant.
- **Demo talking point**: "Feature Gate evaluates numeric and categorical features side by side — no preprocessing needed. It correctly identifies that the bill amount drives tips, not whether someone smokes."

### Dataset 2: Car Fuel Economy

- **Source**: UCI Auto MPG dataset (Carnegie Mellon StatLib), real car specifications from 1970-1973
- **Rows**: 102 vehicles
- **Target**: Miles per gallon
- **Features**: cylinders (numeric), engine_displacement (numeric), horsepower (numeric), vehicle_weight (numeric), acceleration (numeric), model_year (numeric), country_of_origin (categorical), car_name (free text)
- **Demonstrates**: Numeric, categorical, and free-text columns evaluated together; the `car_name` field shows Feature Gate can assess unstructured text
- **Expected results**: `vehicle_weight` and `engine_displacement` should rank highest. `cylinders` and `horsepower` as strong signals. `country_of_origin` (categorical) may show moderate signal. `car_name` (text) is the interesting test — it encodes make/model information that correlates with MPG.
- **Demo talking point**: "Notice that Feature Gate handles a free-text field like car name alongside numeric engine specs — all in one call. No feature engineering required."

### Dataset 3: Airline Passengers (Time Series)

- **Source**: Box & Jenkins (1976), monthly international airline passenger totals 1949-1960
- **Rows**: 144 months
- **Target**: Monthly passenger count
- **Features**: month_number (numeric, time-indexed), year (numeric, time-indexed), trend_index (numeric, time-indexed), month_name (categorical, time-indexed), quarter (categorical, time-indexed)
- **Demonstrates**: Time-series-aware screening with lag scans, block permutations, and autocorrelation-aware null generation
- **Expected results**: `trend_index` and `year` should dominate (the airline industry was growing rapidly). `month_number` should show strong seasonal signal. Check the `best_lag` field — it reveals the seasonal structure.
- **Demo talking point**: "This is where Feature Gate goes beyond standard correlation. It detects that passenger volumes are trending up *and* seasonal — and it reports the lag structure automatically. Standard tests miss this entirely."

### Files Reference

| File | Description |
|------|-------------|
| `demo-datasets/tips.csv` | Raw CSV — restaurant tips data |
| `demo-datasets/mpg.csv` | Raw CSV — car fuel economy data |
| `demo-datasets/airline_passengers.csv` | Raw CSV — monthly airline passenger data |
| `demo-datasets/demo1_tips_payload.json` | Feature Gate API payload for tips |
| `demo-datasets/demo2_mpg_payload.json` | Feature Gate API payload for MPG |
| `demo-datasets/demo3_airline_payload.json` | Feature Gate API payload for airline passengers |
| `demo-datasets/run_demos.py` | Script that sends all three to the API and prints results |

---

## API Reference

### POST /screen

The main endpoint. Accepts a target column and feature columns, runs the full screening pipeline, and returns ranked results.

**Request body:**

```json
{
  "y": {
    "name": "monthly_sales",
    "values": [120.5, 135.2, 128.8, 142.1, 155.3],
    "dtype": "numeric",
    "is_time_indexed": true
  },
  "x_columns": [
    {
      "name": "temperature_avg",
      "values": [18.2, 22.1, 20.5, 25.3, 28.1],
      "dtype": "numeric",
      "is_time_indexed": true
    },
    {
      "name": "promotion_type",
      "values": ["none", "discount", "none", "bogo", "discount"],
      "dtype": "categorical",
      "is_time_indexed": false
    },
    {
      "name": "customer_feedback",
      "values": ["great service", "slow delivery", "love it", "okay", "fantastic"],
      "dtype": "text",
      "is_time_indexed": false
    }
  ],
  "config": {
    "cosine_null_draws": 200,
    "dependence_permutations": 200,
    "fdr_alpha": 0.05,
    "tier2_top_k": 2
  }
}
```

**ColumnPayload fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Column identifier |
| `values` | list | Yes | Column values (numbers, strings, or mixed) |
| `dtype` | string | No | Force type: `numeric`, `categorical`, or `text`. Auto-detected if omitted. |
| `is_time_indexed` | bool | No | Whether values are temporally ordered (default: false) |

**Response body:**

```json
{
  "n_features_screened": 3,
  "n_significant": 1,
  "fdr_alpha": 0.05,
  "tier1_model": "all-MiniLM-L6-v2",
  "tier2_model": "BAAI/bge-base-en-v1.5",
  "results": [
    {
      "name": "temperature_avg",
      "inferred_dtype": "numeric",
      "sketch_preview": "{\"column_name\":\"temperature_avg\",\"max\":28.1,...",
      "cosine_score": 0.8721,
      "cosine_p_value": 0.005,
      "cosine_null_mean": 0.7234,
      "cosine_null_std": 0.0412,
      "hsic_p_value": 0.005,
      "dcor_p_value": 0.005,
      "dcor_statistic": 0.4215,
      "best_lag": 0,
      "best_lag_ccf": 0.6532,
      "combined_p_value": 0.000012,
      "q_value": 0.000036,
      "alignment_score": 12.34,
      "significant": true,
      "tier2_cosine_score": 0.8543,
      "tier2_cosine_p_value": 0.0099
    }
  ],
  "warnings": []
}
```

**Result fields explained:**

| Field | Description |
|---|---|
| `cosine_score` | Raw cosine similarity between feature and target sketch embeddings |
| `cosine_p_value` | Monte Carlo p-value for the cosine score under type-aware nulls |
| `cosine_null_mean/std` | Null distribution statistics (for diagnostics) |
| `hsic_p_value` | HSIC permutation p-value (numeric features only) |
| `dcor_p_value` | Distance correlation permutation p-value (numeric features only) |
| `dcor_statistic` | Distance correlation statistic (0 = independent, 1 = perfectly dependent) |
| `best_lag` | Lag with highest absolute cross-correlation (time-indexed numeric only) |
| `best_lag_ccf` | Cross-correlation at the best lag |
| `combined_p_value` | Fisher-combined p-value across all available tests |
| `q_value` | BH-FDR adjusted p-value (accounts for multiple testing) |
| `alignment_score` | Ranking heuristic (higher = stronger evidence); not a p-value |
| `significant` | Whether q_value < fdr_alpha |
| `tier2_cosine_score/p_value` | Tier-2 model results (only if tier2_top_k is set) |

### GET /health

Returns server and model loading status.

```json
{
  "status": "ok",
  "tier1_loaded": true,
  "tier2_loaded": false
}
```

### GET /config

Returns the full default configuration with all tunable parameters and their descriptions.

---

## Configuration Guide

### Parameter Reference

| Parameter | Default | Range | Description |
|---|---|---|---|
| `cosine_null_draws` | 200 | 50-2000 | Monte Carlo draws per feature. Higher = more precise p-values but slower. Min achievable p-value is ~1/(B+1). |
| `replacement_fraction` | 0.20 | 0.0-1.0 | Fraction of categorical/text values replaced in null generation. Lower for small n or low-cardinality. |
| `token_shuffle_prob` | 0.30 | 0.0-1.0 | Probability of shuffling tokens in the sketch string during null generation. |
| `block_size` | auto | >=1 | Block size for block permutation in time series. Auto-detected based on data length if not set. |
| `enable_ar1_null` | false | bool | Force AR(1) null resampling for numeric time series. Auto-enables if ACF(1) > threshold. |
| `ar1_acf_threshold` | 0.4 | 0.0-1.0 | ACF(1) threshold for auto-enabling AR(1) nulls. |
| `lag_window` | auto | >=0 | Maximum lag for cross-correlation scan. Auto-detected as min(14, n/10). |
| `restrict_positive_lags` | false | bool | Only scan non-negative lags (prevents future leakage). |
| `dependence_permutations` | 200 | 50-2000 | Permutations for HSIC and dCor p-values. |
| `fdr_alpha` | 0.05 | 0.001-0.5 | Significance threshold for BH-FDR. Use 0.10 for exploratory, 0.01 for conservative. |
| `tier1_model` | all-MiniLM-L6-v2 | string | Sentence-transformer model for fast screening. |
| `tier2_model` | BAAI/bge-base-en-v1.5 | string | Higher-quality model for confirmation. |
| `use_onnx` | true | bool | Use ONNX runtime for faster CPU inference. |
| `tier2_top_k` | None | int/None | Number of top features to re-screen with Tier-2. None = skip Tier-2. |
| `sketch_top_k` | 10 | 1-50 | Number of top values to include in categorical sketches. |
| `sketch_budget` | 512 | 128-2048 | Maximum character length for sketch strings. |
| `acf_lags` | [1, 7, 30] | list[int] | Lags for ACF fingerprints in numeric time-series sketches. |
| `random_seed` | 42 | int | Seed for reproducibility. |

### Tuning for Your Domain

**Daily time series (e.g., sales, weather)**:
```json
{
  "block_size": 7,
  "lag_window": 28,
  "acf_lags": [1, 7, 14, 30],
  "restrict_positive_lags": true
}
```

**Hourly time series (e.g., sensor data, traffic)**:
```json
{
  "block_size": 24,
  "lag_window": 72,
  "acf_lags": [1, 24, 168]
}
```

**Large feature store (1000+ columns), exploratory**:
```json
{
  "cosine_null_draws": 100,
  "dependence_permutations": 100,
  "fdr_alpha": 0.10,
  "tier2_top_k": 20
}
```

**Governance/regulatory setting (conservative)**:
```json
{
  "cosine_null_draws": 1000,
  "dependence_permutations": 500,
  "fdr_alpha": 0.01,
  "tier2_top_k": 10
}
```

---

## Understanding the Output

### Reading the Results

Results are returned sorted by `alignment_score` (highest first). The key fields to focus on are:

1. **`significant`**: The bottom-line yes/no after all corrections. If true, this feature has statistically significant evidence of association with Y after controlling for multiple testing.

2. **`q_value`**: The BH-FDR adjusted p-value. Represents the expected false discovery rate if you include all features with q-values at or below this threshold. A q-value of 0.03 means roughly 3% of features at this threshold or below are expected to be false positives.

3. **`alignment_score`**: Use this for ranking. Higher is stronger. Useful for deciding how many features to carry forward into modelling, even among significant ones.

4. **`best_lag`**: If non-zero, the feature may be predictive of Y at a temporal offset. Consider incorporating this lag into your feature engineering.

5. **`cosine_null_mean` and `cosine_null_std`**: Diagnostic fields. If the null mean is very close to the observed cosine with a small null std, the cosine test has low power for this feature — the sketch embedding may not discriminate well.

### Decision Framework

After receiving results, a typical workflow is:

1. **Shortlist**: Take all features where `significant == true`.
2. **Rank**: Order by `alignment_score` to prioritise the strongest signals.
3. **Check lags**: For time-series features, note the `best_lag` and incorporate it into feature engineering.
4. **Tier-2 confirmation**: If `tier2_top_k` was set, compare Tier-1 and Tier-2 cosine scores. Consistent results across both tiers increase confidence.
5. **Investigate borderline features**: Features with q-values in the 0.05-0.15 range may warrant manual review, especially if domain knowledge suggests relevance.
6. **Proceed to modelling**: Use the shortlist as input features for supervised model training, with documented justification from the screening results.

---

## Architecture and Project Structure

```
feature-gate/
|-- main.py                 # FastAPI application, lifespan, endpoints
|-- config.py               # PipelineConfig Pydantic model with all defaults
|-- schemas.py              # Request/response Pydantic schemas
|-- core/
|   |-- __init__.py         # Package exports
|   |-- sketcher.py         # JSON->string sketch construction
|   |                         - _numeric_sketch(): quantiles, moments, ACF
|   |                         - _categorical_sketch(): top-k, entropy, shape
|   |                         - infer_dtype(): auto-detection
|   |                         - build_sketch(): main entry point
|   |
|   |-- embedder.py         # Two-tier frozen transformer management
|   |                         - Lazy model loading with ONNX support
|   |                         - embed_tier1() / embed_tier2()
|   |                         - Normalised to unit vectors
|   |
|   |-- cosine_test.py      # Monte Carlo cosine significance
|   |                         - Numeric nulls: permute, block, AR(1), parametric
|   |                         - Categorical nulls: shuffle + replace
|   |                         - Token shuffle for sketch perturbation
|   |                         - cosine_monte_carlo(): main entry point
|   |
|   |-- dependence.py       # Model-free dependence tests
|   |                         - HSIC with RBF kernels + median heuristic
|   |                         - Distance correlation (double-centred)
|   |                         - Permutation tests (block-aware)
|   |                         - Cross-correlation lag scan
|   |
|   |-- combiner.py         # P-value combination and multiple testing
|   |                         - Fisher's method
|   |                         - Brown's correction for dependent tests
|   |                         - Benjamini-Hochberg FDR
|   |                         - Alignment score heuristic
|   |
|   +-- pipeline.py         # Orchestrator (Algorithm 1)
|                             - Auto-detection of block sizes and lag windows
|                             - Auto-detection of AR(1) necessity
|                             - Batch embedding for efficiency
|                             - Tier-2 confirmation flow
|
|-- example_client.py       # Synthetic data demo (2 signal + 3 noise features)
|-- demo-datasets/          # Real-world demo datasets for client presentations
|   |-- tips.csv            # Restaurant tips (Bryant & Smith 1995)
|   |-- mpg.csv             # Car fuel economy (UCI Auto MPG)
|   |-- airline_passengers.csv  # Monthly passengers (Box & Jenkins 1976)
|   |-- demo1_tips_payload.json
|   |-- demo2_mpg_payload.json
|   |-- demo3_airline_payload.json
|   +-- run_demos.py        # Sends all three to the API and prints results
|-- requirements.txt        # Python dependencies
|-- Dockerfile              # Container deployment with pre-cached model
+-- README.md               # This file
```

---

## Performance Considerations

**Computational bottleneck**: The Monte Carlo cosine test dominates runtime because each null draw requires sketch construction and embedding inference. With B=200 draws across p features, that is 200p embedding calls. At ~15ms per embedding on CPU (MiniLM with ONNX), screening 100 features takes roughly 5 minutes.

**Scaling strategies**:
- Reduce `cosine_null_draws` to 100 for initial exploration (minimum p-value becomes ~0.01).
- Use ONNX backend (enabled by default) for 2-3x speedup over PyTorch.
- HSIC and dCor are O(n^2) in the number of observations — for very long time series (n > 10,000), consider downsampling.
- Tier-2 uses fewer Monte Carlo draws (capped at 100) since it only confirms top-k features.

**Memory**: Peak memory is driven by the embedding model (~100MB for MiniLM, ~400MB for BGE-base) plus the HSIC/dCor kernel matrices (O(n^2) per feature). For n=10,000 observations, each kernel matrix is ~800MB — consider subsetting.

**First request latency**: The Tier-1 model is pre-loaded at startup. The Tier-2 model loads lazily on first use of `tier2_top_k`, adding ~10-20s to the first request that uses it.

---

## What's New in v1.1

Three enhancements addressing the two most significant v1.0 limitations: text signal detection and time series seasonality.

### Token-level text sketching

Text columns now produce token sketches alongside statistical sketches. The sketcher treats each row as a document, scores tokens by `df × log(n_rows / df)` (favouring recurring but non-universal tokens), and includes the top-k tokens with document frequencies in the sketch. This gives the embedder semantic content to work with — a column of product names like "chevrolet impala" now contributes its brand and model tokens rather than just string-shape statistics.

Auto-detection has also improved: multi-token, high-cardinality short strings (product names, item descriptions) are now inferred as `text` even below the 50-character average length threshold.

Control via config:

```json
"config": {
  "text_sketch_mode": "both",     // "statistical" | "tokens" | "both" (default)
  "text_token_top_k": 30
}
```

**Privacy note:** token sketches contain actual vocabulary from the column. For deployments where even column vocabulary must not leave the data layer, set `text_sketch_mode: "statistical"` to restore pure aggregate-statistics sketching (the v1.0 behaviour), at the cost of weaker text signal detection.

The null model for text is token-aware: null draws rebuild a fraction of rows by sampling from the column's own word pool, perturbing document frequency and co-occurrence structure while preserving marginal token statistics. Row shuffling alone would leave the token sketch unchanged and degenerate the test.

### Structural signal reporting

Each numeric time-indexed feature now carries a `has_structure` flag, reported **independently of statistical significance**. Structure is flagged when the lag scan finds a clear peak: `|CCF|` at the best lag exceeds `structure_ccf_threshold` (default 0.30) and stands out from the off-peak background by at least `structure_peak_ratio` (default 1.5×).

This addresses a deliberate conservatism in the significance machinery: autocorrelation-aware nulls correctly raise the significance bar for seasonal features, which means genuinely seasonal features can fall below the q-value threshold. The structural flag surfaces them anyway. A feature with `has_structure: true` but `significant: false` warrants human review — the recommended workflow is Fourier expansion (below).

Calibration (simulation study, N=144 monthly, 200 runs per condition): seasonal features detected 100%, lagged copies 100%, pure noise 0% false positives, pure trend 0% false positives (trend alignment produces uniformly high CCF at all lags, which the peak-dominance test correctly excludes — trend features are already captured by significance).

### Fourier feature expansion

Opt-in synthesis of sin/cos features at user-specified periods, screened alongside your features and clearly tagged as derived:

```json
"config": {
  "fourier_periods": [7, 30.44, 365.25]   // periods in row units
}
```

Suggested periods by sampling granularity: daily data — `[7, 30.44, 91.31, 365.25]`; hourly — `[8, 24, 168]`; monthly — `[3, 12]`. Assumes regular sampling; the Fourier index is row position. Derived features appear in results with `"derived": true` and a `derived_from` generation formula for the audit trail. This turns seasonal alignment into an explicitly testable feature that can attain proper statistical significance, rather than relying on the structural flag alone.

---

## Limitations and Caveats

**Cosine similarity inspects marginal summaries, not joint dependence.** Two features with similar distributional profiles (e.g., both normally distributed with similar means) may produce high cosine scores even without a causal or predictive relationship to Y. The HSIC/dCor tests provide the joint dependence check, but only for numeric features.

**The sketch embedding space was not trained for this task.** The sentence-transformer was trained on natural language similarity, not on statistical sketch comparison. The pipeline relies on the assumption that distributional similarity in sketch text translates to meaningful signal — this works in practice as a fast filter but is not a perfect discriminator.

**Monte Carlo resolution floor.** With B=200, the smallest achievable p-value is ~0.005. You cannot distinguish "strong signal" from "very strong signal" at this resolution. Increase B to 1000+ if fine-grained p-values matter for your use case.

**Non-numeric features get cosine-only testing.** HSIC and dCor require numeric, aligned vectors. Categorical and text features are evaluated solely via the cosine significance test, which provides less statistical power. v1.1's token sketching materially improves text detection, but text screening remains triage rather than a definitive per-token analysis — for token-level attribution, follow up with TF-IDF or similar on shortlisted text columns.

**Independence assumption in Fisher's method.** The three per-feature tests (cosine, HSIC, dCor) are computed on the same data and are not independent. Fisher's method is conservative under positive dependence but can be anti-conservative under negative dependence. Brown's correction (available via the combiner module) addresses this, but is not applied by default.

---

## Troubleshooting

**"Model download is slow or fails"**: The sentence-transformer models are downloaded from Hugging Face on first use. If you are behind a corporate firewall, set `HF_HUB_OFFLINE=1` after downloading models manually, or pre-cache them in the Docker build.

**"All cosine p-values are near 1.0"**: The null distribution may be too similar to the observed distribution. This can happen when the sketch strings for null and observed columns are nearly identical (e.g., permuting a numeric column preserves its marginal statistics exactly). Try increasing `token_shuffle_prob` to 0.4-0.5, or increasing `cosine_null_draws`.

**"HSIC is very slow"**: HSIC computes n x n kernel matrices. For n > 5,000, consider subsampling your data or reducing `dependence_permutations`.

**"OnnxRuntime not found"**: Install it with `pip install onnxruntime`. The pipeline falls back to PyTorch automatically, but ONNX is recommended for CPU performance. If running on Apple Silicon, use `onnxruntime` (not `onnxruntime-gpu`).

**"Length mismatch warning"**: HSIC, dCor, and lag scan require X and Y to have the same number of observations. If they differ (e.g., different sampling frequencies), these tests are skipped and only cosine significance is reported.

# Feature Gate v1.1 Release Notes

Addresses the two most significant v1.0 limitations identified during demo validation: text signal detection (MPG car_name miss) and time series seasonality (airline month_number miss).

## New

**Token-level text sketching** — text columns now include top-k TF-IDF-scored tokens in the sketch, giving the embedder semantic content rather than string-shape statistics alone. New token-aware null generator (row shuffling alone leaves token sketches unchanged; nulls now rebuild rows from the column's word pool). Config: `text_sketch_mode` ("statistical" | "tokens" | "both", default "both"), `text_token_top_k` (default 30). Set `text_sketch_mode: "statistical"` for maximum-privacy deployments (v1.0 behaviour).

**Improved text auto-detection** — multi-token, high-cardinality short strings (product names) now inferred as text below the 50-char threshold.

**Structural signal reporting** — new `has_structure` / `structure_detail` result fields flag clear lag-scan peaks independently of statistical significance. Calibrated thresholds (`structure_ccf_threshold=0.30`, `structure_peak_ratio=1.5`): 100% detection of seasonal/lagged signals, 0% false positives on noise and trend in simulation (N=144 monthly, 200 runs/condition). Response adds `n_with_structure`.

**Fourier feature expansion** — opt-in `fourier_periods` config synthesises sin/cos features at given periods, screened alongside user features, tagged `derived: true` with generation formula in the audit trail.

## API changes (backwards compatible)

- `FeatureResult` adds: `has_structure`, `structure_detail`, `derived`, `derived_from`
- `ScreeningResponse` adds: `n_with_structure`
- `PipelineConfig` adds: `text_sketch_mode`, `text_token_top_k`, `structure_ccf_threshold`, `structure_peak_ratio`, `fourier_periods`

## Validation performed

- Unit tests: token sketch extraction, sketch modes, dtype inference, text null variation, structure detection (peak / flat / trend cases), Fourier synthesis
- Calibration study: structure thresholds against synthetic seasonal, lagged, noise, trend series
- End-to-end pipeline test with Fourier expansion and structure reporting
- Regression: all three demo payloads

## Outstanding for full release validation

- Re-run all three demos against the real embedder (requires local model): confirm car_name is now flagged on MPG, confirm structure/Fourier behaviour on a seasonal dataset
- Categorical group-aware permutations for imbalanced low-cardinality features: deferred to v1.2

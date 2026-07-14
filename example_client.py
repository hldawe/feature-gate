"""
Example client: sends synthetic data to the Feature Gate API
and prints the screening results.

Usage:
    1. Start the server:  python main.py
    2. In another terminal: python example_client.py
"""
import requests
import numpy as np

API_URL = "http://localhost:8000"


def generate_synthetic_data(n: int = 200, seed: int = 42):
    """
    Generate synthetic dataset with known signal and noise features.

    - y:              target (noisy sine wave)
    - x_signal:       correlated numeric (y + noise)
    - x_lagged:       y shifted by 5 steps (lagged signal)
    - x_noise:        pure random noise (should be filtered out)
    - x_categorical:  category correlated with y quantiles
    - x_text_noise:   random text (should be filtered out)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)

    y = np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.3, n)

    x_signal = y + rng.normal(0, 0.5, n)

    x_lagged = np.roll(y, 5) + rng.normal(0, 0.3, n)
    x_lagged[:5] = rng.normal(0, 1, 5)

    x_noise = rng.normal(0, 1, n)

    quartiles = np.percentile(y, [25, 50, 75])
    x_categorical = []
    for val in y:
        if val < quartiles[0]:
            x_categorical.append("low")
        elif val < quartiles[1]:
            x_categorical.append("medium_low")
        elif val < quartiles[2]:
            x_categorical.append("medium_high")
        else:
            x_categorical.append("high")

    fruits = ["apple", "banana", "cherry", "date", "elderberry"]
    x_text_noise = [fruits[int(i)] for i in rng.integers(0, 5, size=n)]

    return {
        "y": {"name": "target_sine", "values": y.tolist()},
        "features": [
            {
                "name": "correlated_signal",
                "values": x_signal.tolist(),
                "is_time_indexed": True,
            },
            {
                "name": "lagged_signal_tau5",
                "values": x_lagged.tolist(),
                "is_time_indexed": True,
            },
            {
                "name": "pure_noise",
                "values": x_noise.tolist(),
                "is_time_indexed": True,
            },
            {
                "name": "categorical_from_y",
                "values": x_categorical,
                "is_time_indexed": False,
            },
            {
                "name": "random_text",
                "values": x_text_noise,
                "is_time_indexed": False,
            },
        ],
    }


def main():
    # Check health
    resp = requests.get(f"{API_URL}/health")
    print(f"Health: {resp.json()}\n")

    # Generate data
    data = generate_synthetic_data(n=200)

    # Build request payload
    payload = {
        "y": {
            "name": data["y"]["name"],
            "values": data["y"]["values"],
        },
        "x_columns": [
            {
                "name": f["name"],
                "values": f["values"],
                "is_time_indexed": f.get("is_time_indexed", False),
            }
            for f in data["features"]
        ],
        "config": {
            "cosine_null_draws": 200,
            "dependence_permutations": 200,
            "fdr_alpha": 0.05,
        },
    }

    print(f"Screening {len(payload['x_columns'])} features against "
          f"'{data['y']['name']}'...")
    print("(This may take a moment on first run while the model loads)\n")

    resp = requests.post(f"{API_URL}/screen", json=payload, timeout=600)

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return

    result = resp.json()

    # Pretty print results
    print("=" * 80)
    print("SCREENING RESULTS")
    print("=" * 80)
    print(f"Features screened: {result['n_features_screened']}")
    print(f"Significant (q < {result['fdr_alpha']}): {result['n_significant']}")
    print(f"Tier-1 model: {result['tier1_model']}")
    print(f"Tier-2 model: {result.get('tier2_model', 'N/A')}")
    if result.get("warnings"):
        print(f"Warnings: {result['warnings']}")
    print()

    for feat in result["results"]:
        sig_marker = " ***" if feat["significant"] else ""
        print(f"--- {feat['name']} ({feat['inferred_dtype']}){sig_marker} ---")
        print(f"  Cosine:    {feat['cosine_score']:.4f}  "
              f"(p={feat['cosine_p_value']:.4f}, "
              f"null={feat['cosine_null_mean']:.4f} "
              f"+/- {feat['cosine_null_std']:.4f})")

        if feat.get("hsic_p_value") is not None:
            print(f"  HSIC p:    {feat['hsic_p_value']:.4f}")
        if feat.get("dcor_p_value") is not None:
            print(f"  dCor:      {feat['dcor_statistic']:.4f}  "
                  f"(p={feat['dcor_p_value']:.4f})")
        if feat.get("best_lag") is not None:
            print(f"  Best lag:  {feat['best_lag']}  "
                  f"(CCF={feat['best_lag_ccf']:.4f})")
        if feat.get("tier2_cosine_score") is not None:
            print(f"  Tier-2:    {feat['tier2_cosine_score']:.4f}  "
                  f"(p={feat['tier2_cosine_p_value']:.4f})")

        print(f"  Combined:  p={feat['combined_p_value']:.6f}  "
              f"q={feat['q_value']:.6f}  "
              f"align={feat['alignment_score']:.2f}")
        print()


if __name__ == "__main__":
    main()

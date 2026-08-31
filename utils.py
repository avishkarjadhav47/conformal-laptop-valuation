"""
App-side helpers.
Turns raw, human-typed specs into a model-ready row, gets a point prediction
plus a calibrated 80% prediction interval, explains the prediction (perturbation-based
local feature contribution — a from-scratch stand-in for SHAP), and scores
a real listing against the model's estimate.
"""

import numpy as np
import pandas as pd

pd.set_option("future.infer_string", False)

from feature_engineering import cpu_features, processor_features, gpu_features, os_family, calculate_ppi

NUM_COLS = ["Ram", "ROM", "warranty", "ppi", "Total_Cores", "Threads",
            "P_Cores", "E_Cores", "Threads_per_Core", "Generation", "VRAM_GB"]
CAT_COLS = ["brand", "Ram_type", "Core_Type", "Processor_Brand", "Processor_Family",
            "GPU_Brand", "GPU_Family", "OS_Family"]
BIN_COLS = ["Hybrid_CPU", "Hyperthreading", "Integrated_GPU", "Dedicated_GPU", "SSD"]

# Friendly labels for the explanation panel
FRIENDLY_NAMES = {
    "Ram": "RAM capacity",
    "ROM": "Storage capacity",
    "warranty": "Warranty length",
    "ppi": "Screen sharpness (PPI)",
    "Total_Cores": "CPU core count",
    "Threads": "CPU thread count",
    "P_Cores": "Performance cores",
    "E_Cores": "Efficiency cores",
    "Threads_per_Core": "Threads per core",
    "Generation": "CPU generation",
    "VRAM_GB": "GPU memory (VRAM)",
    "brand": "Brand",
    "Ram_type": "RAM type",
    "Core_Type": "CPU core type",
    "Processor_Brand": "Processor brand",
    "Processor_Family": "Processor family",
    "GPU_Brand": "GPU brand",
    "GPU_Family": "GPU family",
    "OS_Family": "Operating system",
    "Hybrid_CPU": "Hybrid (P+E core) CPU",
    "Hyperthreading": "Hyperthreading",
    "Integrated_GPU": "Integrated graphics",
    "Dedicated_GPU": "Dedicated graphics",
    "SSD": "SSD storage",
}


def build_feature_row(
    brand, processor_text, cpu_text, ram_gb, ram_type, rom_gb, rom_type,
    gpu_text, warranty, screen_size, resolution, os_text,
):
    """Turn raw, user-supplied spec strings into the engineered feature dict the model was trained on."""
    ppi = calculate_ppi(screen_size, resolution)

    (total_cores, threads, p_cores, e_cores, hybrid_cpu,
     core_type, hyperthreading, threads_per_core) = cpu_features(cpu_text)

    processor_brand, processor_family, generation = processor_features(processor_text)

    (gpu_brand, gpu_family, vram, integrated_gpu, dedicated_gpu) = gpu_features(gpu_text)

    ssd = 1 if rom_type == "SSD" else 0
    os_fam = os_family(os_text)

    row = {
        "brand": brand,
        "Ram": ram_gb,
        "Ram_type": ram_type,
        "ROM": rom_gb,
        "warranty": warranty,
        "ppi": ppi,
        "Total_Cores": total_cores,
        "Threads": threads,
        "P_Cores": p_cores,
        "E_Cores": e_cores,
        "Hybrid_CPU": hybrid_cpu,
        "Core_Type": core_type,
        "Hyperthreading": hyperthreading,
        "Threads_per_Core": threads_per_core,
        "Processor_Brand": processor_brand,
        "Processor_Family": processor_family,
        "Generation": generation,
        "GPU_Brand": gpu_brand,
        "GPU_Family": gpu_family,
        "VRAM_GB": vram,
        "Integrated_GPU": integrated_gpu,
        "Dedicated_GPU": dedicated_gpu,
        "SSD": ssd,
        "OS_Family": os_fam,
    }
    return row


def to_model_frame(row: dict, model_columns: list) -> pd.DataFrame:
    return pd.DataFrame([row])[model_columns]


def predict_with_interval(model, conformal_quantile, input_df: pd.DataFrame):
    """
    Point estimate plus an 80% prediction interval from a SINGLE model.

    Split-conformal validity requires the interval to be centred on the same
    model whose calibration residuals produced `conformal_quantile`. Centring
    it on a different (better-fit) model would void the coverage guarantee,
    so the point estimate and both bounds all come from `model`.

    The interval is symmetric in log space, so on the rupee scale it is
    asymmetric around the point estimate. That is expected, not a bug.
    """
    point_log = float(model.predict(input_df)[0])

    point = float(np.expm1(point_log))
    lo = float(np.expm1(point_log - conformal_quantile))
    hi = float(np.expm1(point_log + conformal_quantile))

    # expm1 is monotonic, so lo < point < hi holds by construction -- no clamps.
    return point, max(0.0, lo), hi


def explain_prediction(model, input_df: pd.DataFrame, reference: dict, top_n: int = 5):
    """
    Perturbation-based local explanation: for each feature, swap in the
    "typical" (median/mode) value from the training set and measure how
    much the prediction moves. A large positive delta means that feature is
    pushing the price up relative to a typical laptop; negative means it's
    pulling the price down. This is a lightweight, dependency-free stand-in
    for SHAP — same idea (marginal contribution of a feature), simpler
    estimator (one-at-a-time swap instead of averaging over coalitions).
    """
    base_pred = float(np.expm1(model.predict(input_df)[0]))
    contributions = []

    for col in input_df.columns:
        if col not in reference:
            continue
        perturbed = input_df.copy()
        perturbed.at[perturbed.index[0], col] = reference[col]
        perturbed_pred = float(np.expm1(model.predict(perturbed)[0]))
        delta = base_pred - perturbed_pred  # how much this feature's actual value adds vs. "typical"
        if abs(delta) > 1:  # ignore near-zero noise
            contributions.append((col, delta))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:top_n]
    return [
        {"feature": FRIENDLY_NAMES.get(col, col), "impact_inr": round(delta)}
        for col, delta in top
    ]


def deal_score(predicted_price: float, listed_price: float):
    """
    Compare a real listing price to the model's fair-price estimate.
    Positive pct_diff = listed above model estimate (potentially overpriced).
    Negative pct_diff = listed below model estimate (potentially a good deal).
    """
    pct_diff = ((listed_price - predicted_price) / predicted_price) * 100

    if pct_diff <= -10:
        label, tone = "Good deal — priced well below estimate", "good"
    elif pct_diff <= -3:
        label, tone = "Slightly below estimate", "good"
    elif pct_diff < 3:
        label, tone = "Fairly priced", "neutral"
    elif pct_diff < 10:
        label, tone = "Slightly above estimate", "warn"
    else:
        label, tone = "Overpriced — well above estimate", "bad"

    return {"pct_diff": round(pct_diff, 1), "label": label, "tone": tone}

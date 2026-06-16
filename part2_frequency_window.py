# ============================================================
# Part 2: Frequency-Window Analysis with Corrected SLS Time Constant
# 100-point standard version
#
# Core correction:
# tau_ref = eta / E2, not eta / E1
#
# Purpose:
# 1. Define a dimensionless loading-frequency framework.
# 2. Separate mechanical response from phenomenological biological filtering.
# 3. Show ablation explicitly so the finite window is not overclaimed.
# 4. Treat cellular cutoff as a sensitivity parameter, not a measured constant.
# 5. Export individual 5:3 PNG panels, CSV tables, and a 600 dpi combined overview.
#
# Author: Chen Zong
# ============================================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

np.random.seed(42)

# ============================================================
# 0. Output folder and style
# ============================================================

OUTDIR = "part2_frequency_window_outputs"
os.makedirs(OUTDIR, exist_ok=True)

DPI = 300
FIGSIZE = (5, 3)
BG = "#FAFAFA"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelcolor": "#2d2d2d",
    "axes.edgecolor": "#999999",
    "axes.linewidth": 0.8,
    "xtick.color": "#2d2d2d",
    "ytick.color": "#2d2d2d",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "savefig.dpi": DPI,
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "mechanical": "#7B9EA6",
    "cell": "#B5A07A",
    "risk": "#A67B7B",
    "net": "#4A6572",
    "phase": "#8A9E82",
    "dissipation": "#8E85A6",
    "window": "#A67B7B",
    "gray": "#777777",
    "mechanical_only": "#7B9EA6",
    "cell_only": "#B5A07A",
    "risk_only": "#8A9E82",
    "full": "#4A6572",
}

CMAP_SEQ = mcolors.LinearSegmentedColormap.from_list(
    "morandi_seq",
    ["#EEF3F4", "#C8D8DF", "#7B9EA6", "#4A6572", "#A67B7B"]
)

# ============================================================
# 1. Baseline parameters
# ============================================================

BASE_PARAMS = {
    "E1": 0.5,
    "E2": 2.0,
    "eta": 10.0,
}

TARGET_RMS_STRESS = 1.0
PRIMARY_MEAN_STRESS_FRACTION = 0.60

OMEGA_TAU_GRID = np.logspace(-4, 5, 1400)
OMEGA_TAU_DESIGN_GRID = np.logspace(-2, 3, 160)

MEAN_FRACTION_GRID = np.array([0.30, 0.45, 0.60, 0.75, 0.90])

WINDOW_LEVEL = 0.90

# This is not a measured biological constant.
# These are scenario values for sensitivity analysis.
CELLULAR_CUTOFF_SCENARIOS = [20.0, 40.0, 80.0, 160.0, 320.0]
REFERENCE_CELLULAR_CUTOFF = 80.0
CELLULAR_FILTER_HILL = 2.0

HIGH_FREQ_CENTER_SCENARIOS = [60.0, 120.0, 240.0, 480.0]
REFERENCE_HIGH_FREQ_CENTER = 120.0
HIGH_FREQ_HILL = 2.0

N_MC = 2000

# Conservative translational scaling range.
TAU_SCALING_SECONDS = np.logspace(0, 2, 160)

EPS = 1e-12

# ============================================================
# 2. Helpers
# ============================================================

def save_fig(fig, filename):
    fig.set_size_inches(FIGSIZE[0], FIGSIZE[1], forward=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(OUTDIR, filename), dpi=DPI, facecolor=BG)
    plt.close(fig)

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(labelsize=7)

def panel_label(ax, label):
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#2d2d2d",
        va="top",
        ha="left"
    )

def minmax_normalize(x):
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if xmax - xmin < EPS:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def robust_normalize(x, lower_q=0.02, upper_q=0.98):
    x = np.asarray(x, dtype=float)
    lo = np.nanquantile(x, lower_q)
    hi = np.nanquantile(x, upper_q)
    if hi - lo < EPS:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

def tau_ref_from_params(E1, E2, eta):
    """
    Corrected SLS Maxwell-branch reference time constant.

    For the standard linear solid used here, the Maxwell branch contains
    E2 and eta. Therefore the reference relaxation time used for
    dimensionless scaling is eta / E2.

    This is a model reference time constant, not a measured PDL constant.
    """
    return eta / E2

def frequency_from_omega_tau(omega_tau, E1, E2, eta):
    tau_ref = tau_ref_from_params(E1, E2, eta)
    return np.asarray(omega_tau, dtype=float) / (2.0 * np.pi * tau_ref)

def period_from_omega_tau(omega_tau, E1, E2, eta):
    freq = frequency_from_omega_tau(omega_tau, E1, E2, eta)
    return 1.0 / np.maximum(freq, EPS)

def cellular_filter(omega_tau, cutoff, hill=CELLULAR_FILTER_HILL):
    """
    Phenomenological low-pass cellular-response scenario.

    cutoff is not treated as a known biological constant.
    All conclusions using this filter must be interpreted through
    sensitivity analysis.
    """
    omega_tau = np.asarray(omega_tau, dtype=float)
    return 1.0 / (1.0 + (omega_tau / cutoff) ** hill)

def high_frequency_penalty(omega_tau, center, hill=HIGH_FREQ_HILL):
    """
    Phenomenological high-frequency penalty scenario.

    This is not a measured injury curve.
    """
    omega_tau = np.asarray(omega_tau, dtype=float)
    x = (omega_tau / center) ** hill
    return x / (1.0 + x)

def complex_compliance_sls(freq_hz, E1, E2, eta):
    """
    Frequency-domain compliance of the standard linear solid.

    J*(omega) = (1 + i omega eta / E2)
                /
                (E1 + i omega eta * (1 + E1/E2))

    This expression preserves the full E1, E2, eta dependence.
    Dimensionless scaling uses tau_ref = eta/E2.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    omega = 2.0 * np.pi * freq_hz

    J = np.zeros_like(omega, dtype=complex)

    zero_mask = omega < EPS
    nonzero_mask = ~zero_mask

    J[zero_mask] = 1.0 / E1

    numerator = 1.0 + 1j * omega[nonzero_mask] * eta / E2
    denominator = E1 + 1j * omega[nonzero_mask] * eta * (1.0 + E1 / E2)

    J[nonzero_mask] = numerator / denominator

    return J

def find_internal_optimum_status(x_best, grid):
    grid = np.asarray(grid, dtype=float)
    if x_best <= grid.min() * 1.05:
        return "Lower boundary"
    if x_best >= grid.max() / 1.05:
        return "Upper boundary"
    return "Internal optimum"

def summarize_window(df, score_col, level=WINDOW_LEVEL):
    idx = int(df[score_col].idxmax())
    best = df.loc[idx]

    window_df = df[df[score_col] >= level].copy()

    if len(window_df) > 0:
        low = float(window_df["Omega_tau"].min())
        high = float(window_df["Omega_tau"].max())
        low_hz = float(window_df["Frequency_Hz"].min())
        high_hz = float(window_df["Frequency_Hz"].max())
    else:
        low = np.nan
        high = np.nan
        low_hz = np.nan
        high_hz = np.nan

    return {
        "Best_omega_tau": float(best["Omega_tau"]),
        "Best_frequency_Hz": float(best["Frequency_Hz"]),
        "Best_period_seconds": 1.0 / max(float(best["Frequency_Hz"]), EPS),
        "Window_level": level,
        "Window_low_omega_tau": low,
        "Window_high_omega_tau": high,
        "Window_low_frequency_Hz": low_hz,
        "Window_high_frequency_Hz": high_hz,
        "Window_short_period_seconds": 1.0 / max(high_hz, EPS) if np.isfinite(high_hz) else np.nan,
        "Window_long_period_seconds": 1.0 / max(low_hz, EPS) if np.isfinite(low_hz) else np.nan,
        "Optimum_location_status": find_internal_optimum_status(float(best["Omega_tau"]), df["Omega_tau"].values),
    }

# ============================================================
# 3. Core frequency-response model
# ============================================================

def compute_frequency_response(
    omega_tau_grid,
    E1,
    E2,
    eta,
    mean_fraction=PRIMARY_MEAN_STRESS_FRACTION,
    target_rms_stress=TARGET_RMS_STRESS,
    cellular_cutoff=REFERENCE_CELLULAR_CUTOFF,
    high_freq_center=REFERENCE_HIGH_FREQ_CENTER,
    mode="full"
):
    """
    mode options:
    - "mechanical_only": no cellular filter, no overload penalty
    - "cell_filter_only": cellular low-pass filter, no overload penalty
    - "overload_only": no cellular filter, overload penalty included
    - "full": cellular filter and overload penalty included

    Important:
    Only the full model can generate a finite candidate biological window.
    Mechanical-only response should not be overinterpreted as a biological optimum.
    """
    omega_tau_grid = np.asarray(omega_tau_grid, dtype=float)

    mean_stress = target_rms_stress * mean_fraction
    dynamic_amp = np.sqrt(max(target_rms_stress ** 2 - mean_stress ** 2, EPS))

    freq_hz = frequency_from_omega_tau(omega_tau_grid, E1, E2, eta)
    omega = 2.0 * np.pi * freq_hz

    J = complex_compliance_sls(freq_hz, E1, E2, eta)

    strain_amp = dynamic_amp * np.abs(J)
    strain_rate_amp = omega * strain_amp
    phase_lag_degree = np.abs(np.degrees(np.angle(J)))
    dissipation_proxy = np.pi * dynamic_amp ** 2 * np.abs(np.imag(J))

    strain_norm = minmax_normalize(strain_amp)
    strain_rate_norm = minmax_normalize(strain_rate_amp)
    phase_norm = minmax_normalize(phase_lag_degree)
    dissipation_norm = minmax_normalize(dissipation_proxy)
    fatigue_norm = minmax_normalize(omega_tau_grid * strain_amp)

    cell = cellular_filter(
        omega_tau_grid,
        cutoff=cellular_cutoff,
        hill=CELLULAR_FILTER_HILL
    )

    high_cost = high_frequency_penalty(
        omega_tau_grid,
        center=high_freq_center,
        hill=HIGH_FREQ_HILL
    )

    mechanical_drive = np.sqrt(
        np.maximum(strain_rate_norm, 0.0) *
        np.maximum(phase_norm, 0.0)
    )

    if mode in ["mechanical_only", "overload_only"]:
        applied_cell_filter = np.ones_like(cell)
    else:
        applied_cell_filter = cell

    filtered_drive = mechanical_drive * applied_cell_filter

    overload_penalty = (
        0.25 * strain_norm +
        0.20 * dissipation_norm +
        0.30 * high_cost +
        0.25 * fatigue_norm
    )

    if mode in ["mechanical_only", "cell_filter_only"]:
        applied_overload_penalty = np.zeros_like(overload_penalty)
    else:
        applied_overload_penalty = overload_penalty

    safety = np.maximum(1.0 - applied_overload_penalty, 0.0)

    raw_score = filtered_drive * safety
    normalized_score = minmax_normalize(raw_score)

    return pd.DataFrame({
        "Omega_tau": omega_tau_grid,
        "Frequency_Hz": freq_hz,
        "Period_seconds": 1.0 / np.maximum(freq_hz, EPS),
        "Mean_stress_fraction": mean_fraction,
        "Mean_stress": mean_stress,
        "Dynamic_stress_amplitude": dynamic_amp,
        "Strain_amplitude": strain_amp,
        "Strain_rate_amplitude": strain_rate_amp,
        "Phase_lag_degree": phase_lag_degree,
        "Dissipation_proxy": dissipation_proxy,
        "Strain_norm": strain_norm,
        "Strain_rate_norm": strain_rate_norm,
        "Phase_norm": phase_norm,
        "Dissipation_norm": dissipation_norm,
        "Fatigue_norm": fatigue_norm,
        "Cellular_filter_raw": cell,
        "Cellular_filter_applied": applied_cell_filter,
        "High_frequency_cost": high_cost,
        "Mechanical_drive": mechanical_drive,
        "Filtered_drive": filtered_drive,
        "Overload_penalty_raw": overload_penalty,
        "Overload_penalty_applied": applied_overload_penalty,
        "Safety_factor": safety,
        "Raw_score": raw_score,
        "Normalized_score": normalized_score,
        "Mode": mode,
        "Cellular_cutoff_omega_tau": cellular_cutoff,
        "High_frequency_penalty_center_omega_tau": high_freq_center,
    })

# ============================================================
# 4. Baseline response and ablation
# ============================================================

print("Running Part 2 frequency-window analysis with tau_ref = eta / E2...")

E1 = BASE_PARAMS["E1"]
E2 = BASE_PARAMS["E2"]
eta = BASE_PARAMS["eta"]

TAU_REF = tau_ref_from_params(E1, E2, eta)

assert abs(TAU_REF - eta / E2) < EPS
assert abs(TAU_REF - eta / E1) > EPS, "Wrong tau definition detected."

modes = ["mechanical_only", "cell_filter_only", "overload_only", "full"]

mode_labels = {
    "mechanical_only": "Mechanical only",
    "cell_filter_only": "Cell filter only",
    "overload_only": "Overload only",
    "full": "Full phenomenological model",
}

mode_colors = {
    "mechanical_only": COLORS["mechanical_only"],
    "cell_filter_only": COLORS["cell_only"],
    "overload_only": COLORS["risk_only"],
    "full": COLORS["full"],
}

df_mode_list = []
summary_records = []

for mode in modes:
    tmp = compute_frequency_response(
        omega_tau_grid=OMEGA_TAU_GRID,
        E1=E1,
        E2=E2,
        eta=eta,
        mean_fraction=PRIMARY_MEAN_STRESS_FRACTION,
        cellular_cutoff=REFERENCE_CELLULAR_CUTOFF,
        high_freq_center=REFERENCE_HIGH_FREQ_CENTER,
        mode=mode
    )
    df_mode_list.append(tmp)

    s = summarize_window(tmp, "Normalized_score", level=WINDOW_LEVEL)
    s.update({
        "Mode": mode,
        "Mode_label": mode_labels[mode],
        "E1": E1,
        "E2": E2,
        "eta": eta,
        "Tau_ref_eta_over_E2_seconds": TAU_REF,
        "Cellular_cutoff_omega_tau": REFERENCE_CELLULAR_CUTOFF,
        "High_frequency_penalty_center_omega_tau": REFERENCE_HIGH_FREQ_CENTER,
    })
    summary_records.append(s)

df_modes = pd.concat(df_mode_list, ignore_index=True)
df_ablation_summary = pd.DataFrame(summary_records)

df_modes.to_csv(os.path.join(OUTDIR, "part2_v3_frequency_response_all_ablation_modes.csv"), index=False)
df_ablation_summary.to_csv(os.path.join(OUTDIR, "part2_v3_ablation_summary.csv"), index=False)

df_full = df_modes[df_modes["Mode"] == "full"].copy()
full_summary = df_ablation_summary[df_ablation_summary["Mode"] == "full"].iloc[0].to_dict()

BEST_OMEGA_TAU = float(full_summary["Best_omega_tau"])
WINDOW_LOW = float(full_summary["Window_low_omega_tau"])
WINDOW_HIGH = float(full_summary["Window_high_omega_tau"])

baseline_summary = pd.DataFrame([{
    "Best_omega_tau": BEST_OMEGA_TAU,
    "Best_frequency_Hz": float(full_summary["Best_frequency_Hz"]),
    "Best_period_seconds": float(full_summary["Best_period_seconds"]),
    "Window_level": WINDOW_LEVEL,
    "Window_low_omega_tau": WINDOW_LOW,
    "Window_high_omega_tau": WINDOW_HIGH,
    "Window_low_frequency_Hz": float(full_summary["Window_low_frequency_Hz"]),
    "Window_high_frequency_Hz": float(full_summary["Window_high_frequency_Hz"]),
    "Window_short_period_seconds": float(full_summary["Window_short_period_seconds"]),
    "Window_long_period_seconds": float(full_summary["Window_long_period_seconds"]),
    "Optimum_location_status": full_summary["Optimum_location_status"],
    "E1": E1,
    "E2": E2,
    "eta": eta,
    "Tau_ref_definition": "eta / E2",
    "Tau_ref_eta_over_E2_seconds": TAU_REF,
    "Cellular_cutoff_omega_tau": REFERENCE_CELLULAR_CUTOFF,
    "High_frequency_penalty_center_omega_tau": REFERENCE_HIGH_FREQ_CENTER,
    "Interpretation": "Model-derived candidate window under a phenomenological full model; not a measured PDL constant."
}])

baseline_summary.to_csv(os.path.join(OUTDIR, "part2_v3_baseline_full_model_window_summary.csv"), index=False)

# ============================================================
# 5. Cellular cutoff sensitivity
# ============================================================

cutoff_records = []
cutoff_curve_records = []

for cutoff in CELLULAR_CUTOFF_SCENARIOS:
    tmp = compute_frequency_response(
        omega_tau_grid=OMEGA_TAU_GRID,
        E1=E1,
        E2=E2,
        eta=eta,
        mean_fraction=PRIMARY_MEAN_STRESS_FRACTION,
        cellular_cutoff=cutoff,
        high_freq_center=REFERENCE_HIGH_FREQ_CENTER,
        mode="full"
    )

    s = summarize_window(tmp, "Normalized_score", level=WINDOW_LEVEL)
    s.update({
        "Cellular_cutoff_omega_tau": cutoff,
        "High_frequency_penalty_center_omega_tau": REFERENCE_HIGH_FREQ_CENTER,
        "Tau_ref_eta_over_E2_seconds": TAU_REF,
    })
    cutoff_records.append(s)

    keep = tmp[["Omega_tau", "Normalized_score"]].copy()
    keep["Cellular_cutoff_omega_tau"] = cutoff
    cutoff_curve_records.append(keep)

df_cutoff_summary = pd.DataFrame(cutoff_records)
df_cutoff_curves = pd.concat(cutoff_curve_records, ignore_index=True)

df_cutoff_summary.to_csv(os.path.join(OUTDIR, "part2_v3_cellular_cutoff_sensitivity_summary.csv"), index=False)
df_cutoff_curves.to_csv(os.path.join(OUTDIR, "part2_v3_cellular_cutoff_sensitivity_curves.csv"), index=False)

# ============================================================
# 6. High-frequency penalty sensitivity
# ============================================================

hf_records = []
hf_curve_records = []

for center in HIGH_FREQ_CENTER_SCENARIOS:
    tmp = compute_frequency_response(
        omega_tau_grid=OMEGA_TAU_GRID,
        E1=E1,
        E2=E2,
        eta=eta,
        mean_fraction=PRIMARY_MEAN_STRESS_FRACTION,
        cellular_cutoff=REFERENCE_CELLULAR_CUTOFF,
        high_freq_center=center,
        mode="full"
    )

    s = summarize_window(tmp, "Normalized_score", level=WINDOW_LEVEL)
    s.update({
        "Cellular_cutoff_omega_tau": REFERENCE_CELLULAR_CUTOFF,
        "High_frequency_penalty_center_omega_tau": center,
        "Tau_ref_eta_over_E2_seconds": TAU_REF,
    })
    hf_records.append(s)

    keep = tmp[["Omega_tau", "Normalized_score"]].copy()
    keep["High_frequency_penalty_center_omega_tau"] = center
    hf_curve_records.append(keep)

df_hf_summary = pd.DataFrame(hf_records)
df_hf_curves = pd.concat(hf_curve_records, ignore_index=True)

df_hf_summary.to_csv(os.path.join(OUTDIR, "part2_v3_high_frequency_penalty_sensitivity_summary.csv"), index=False)
df_hf_curves.to_csv(os.path.join(OUTDIR, "part2_v3_high_frequency_penalty_sensitivity_curves.csv"), index=False)

# ============================================================
# 7. Design-space over mean stress fraction
# ============================================================

design_records = []

for mean_fraction in MEAN_FRACTION_GRID:
    tmp = compute_frequency_response(
        omega_tau_grid=OMEGA_TAU_DESIGN_GRID,
        E1=E1,
        E2=E2,
        eta=eta,
        mean_fraction=mean_fraction,
        cellular_cutoff=REFERENCE_CELLULAR_CUTOFF,
        high_freq_center=REFERENCE_HIGH_FREQ_CENTER,
        mode="full"
    )
    design_records.append(tmp)

df_design = pd.concat(design_records, ignore_index=True)
df_design.to_csv(os.path.join(OUTDIR, "part2_v3_design_space_mean_load.csv"), index=False)

# ============================================================
# 8. Monte Carlo robustness
# ============================================================

mc_records = []

for i in range(N_MC):
    E1_mc = 10 ** np.random.uniform(np.log10(0.05), np.log10(5.0))
    E2_mc = E1_mc * 10 ** np.random.uniform(np.log10(2.0), np.log10(10.0))
    eta_mc = 10 ** np.random.uniform(np.log10(1.0), np.log10(200.0))

    cutoff_mc = 10 ** np.random.uniform(np.log10(20.0), np.log10(320.0))
    hf_center_mc = 10 ** np.random.uniform(np.log10(60.0), np.log10(480.0))
    mean_mc = np.random.uniform(0.45, 0.75)

    tmp = compute_frequency_response(
        omega_tau_grid=np.logspace(-2, 3, 280),
        E1=E1_mc,
        E2=E2_mc,
        eta=eta_mc,
        mean_fraction=mean_mc,
        cellular_cutoff=cutoff_mc,
        high_freq_center=hf_center_mc,
        mode="full"
    )

    s = summarize_window(tmp, "Normalized_score", level=WINDOW_LEVEL)

    mc_records.append({
        "Iteration": i + 1,
        "E1": E1_mc,
        "E2": E2_mc,
        "eta": eta_mc,
        "Tau_ref_eta_over_E2_seconds": tau_ref_from_params(E1_mc, E2_mc, eta_mc),
        "Mean_stress_fraction": mean_mc,
        "Cellular_cutoff_omega_tau": cutoff_mc,
        "High_frequency_penalty_center_omega_tau": hf_center_mc,
        "Best_omega_tau": s["Best_omega_tau"],
        "Best_frequency_Hz": s["Best_frequency_Hz"],
        "Best_inside_baseline_90pct_window": bool((s["Best_omega_tau"] >= WINDOW_LOW) and (s["Best_omega_tau"] <= WINDOW_HIGH)),
        "Window_low_omega_tau": s["Window_low_omega_tau"],
        "Window_high_omega_tau": s["Window_high_omega_tau"],
        "Optimum_location_status": s["Optimum_location_status"],
    })

df_mc = pd.DataFrame(mc_records)

df_mc.to_csv(os.path.join(OUTDIR, "part2_v3_monte_carlo_robustness.csv"), index=False)

mc_summary = pd.DataFrame([{
    "N": N_MC,
    "Median_best_omega_tau": float(df_mc["Best_omega_tau"].median()),
    "Q1_best_omega_tau": float(df_mc["Best_omega_tau"].quantile(0.25)),
    "Q3_best_omega_tau": float(df_mc["Best_omega_tau"].quantile(0.75)),
    "P05_best_omega_tau": float(df_mc["Best_omega_tau"].quantile(0.05)),
    "P95_best_omega_tau": float(df_mc["Best_omega_tau"].quantile(0.95)),
    "Fraction_inside_baseline_90pct_window": float(df_mc["Best_inside_baseline_90pct_window"].mean()),
    "Internal_optimum_fraction": float((df_mc["Optimum_location_status"] == "Internal optimum").mean()),
}])

mc_summary.to_csv(os.path.join(OUTDIR, "part2_v3_monte_carlo_summary.csv"), index=False)

# ============================================================
# 9. Absolute frequency scaling, tau 1-100 s
# ============================================================

scaling_records = []

for tau_seconds in TAU_SCALING_SECONDS:
    scaling_records.append({
        "Measured_or_assumed_tau_seconds": tau_seconds,
        "Best_omega_tau": BEST_OMEGA_TAU,
        "Window_low_omega_tau": WINDOW_LOW,
        "Window_high_omega_tau": WINDOW_HIGH,
        "Best_frequency_Hz": BEST_OMEGA_TAU / (2.0 * np.pi * tau_seconds),
        "Window_low_frequency_Hz": WINDOW_LOW / (2.0 * np.pi * tau_seconds),
        "Window_high_frequency_Hz": WINDOW_HIGH / (2.0 * np.pi * tau_seconds),
        "Best_period_seconds": (2.0 * np.pi * tau_seconds) / max(BEST_OMEGA_TAU, EPS),
        "Window_short_period_seconds": (2.0 * np.pi * tau_seconds) / max(WINDOW_HIGH, EPS),
        "Window_long_period_seconds": (2.0 * np.pi * tau_seconds) / max(WINDOW_LOW, EPS),
    })

df_scaling = pd.DataFrame(scaling_records)

df_scaling.to_csv(os.path.join(OUTDIR, "part2_v3_absolute_frequency_scaling_tau_1_to_100s.csv"), index=False)

# ============================================================
# 10. Decision and interpretation_note tables
# ============================================================

decision_table = pd.DataFrame([
    {
        "Decision_domain": "Reference time constant",
        "Result": "tau_ref = eta / E2",
        "Interpretation": "Corrected SLS Maxwell-branch time constant. No eta/E1 is used anywhere."
    },
    {
        "Decision_domain": "Mechanical-only result",
        "Result": "Reported separately",
        "Interpretation": "Mechanical-only response is not treated as a biological optimum."
    },
    {
        "Decision_domain": "Full model window",
        "Result": f"Best omega_tau = {BEST_OMEGA_TAU:.4g}; 90% window = {WINDOW_LOW:.4g} to {WINDOW_HIGH:.4g}",
        "Interpretation": "This is a model-derived candidate window under phenomenological filtering and risk assumptions."
    },
    {
        "Decision_domain": "Cellular cutoff",
        "Result": f"Sensitivity tested from {min(CELLULAR_CUTOFF_SCENARIOS):.4g} to {max(CELLULAR_CUTOFF_SCENARIOS):.4g}",
        "Interpretation": "The cutoff is not claimed as a measured PDL constant."
    },
    {
        "Decision_domain": "High-frequency penalty",
        "Result": f"Sensitivity tested from {min(HIGH_FREQ_CENTER_SCENARIOS):.4g} to {max(HIGH_FREQ_CENTER_SCENARIOS):.4g}",
        "Interpretation": "High-frequency penalty is a risk scenario, not a measured injury curve."
    },
    {
        "Decision_domain": "Absolute frequency",
        "Result": "f = omega_tau / (2*pi*tau)",
        "Interpretation": "Absolute Hz requires measured tissue or construct tau."
    },
])

decision_table.to_csv(os.path.join(OUTDIR, "part2_v3_decision_table.csv"), index=False)

interpretation_note_table = pd.DataFrame([
    {
        "Rule": "Never use eta/E1 for the SLS reference time constant",
        "Meaning": "The reference definition is tau_ref = eta/E2."
    },
    {
        "Rule": "Do not claim measured PDL optimum",
        "Meaning": "The predicted window is a model-derived hypothesis."
    },
    {
        "Rule": "Do not claim universal absolute frequency",
        "Meaning": "Hz must be scaled by measured tau."
    },
    {
        "Rule": "Cellular filter is phenomenological",
        "Meaning": "The cutoff must be supported by sensitivity analysis or future experiments."
    },
    {
        "Rule": "Ablation is part of the result",
        "Meaning": "Mechanical-only and full-model curves must be shown together."
    },
    {
        "Rule": "Part 2 is not primary waveform ranking",
        "Meaning": "Waveform ranking belongs to Part 3."
    },
])

interpretation_note_table.to_csv(os.path.join(OUTDIR, "part2_v3_interpretation_interpretation_notes.csv"), index=False)

# ============================================================
# 11. Figures A-J
# ============================================================

# A. Frequency response components
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

ax.plot(df_full["Omega_tau"], df_full["Mechanical_drive"], color=COLORS["mechanical"], linewidth=1.5, label="Mechanical drive")
ax.plot(df_full["Omega_tau"], df_full["Cellular_filter_raw"], color=COLORS["cell"], linewidth=1.4, label="Cellular filter")
ax.plot(df_full["Omega_tau"], df_full["Overload_penalty_raw"], color=COLORS["risk"], linewidth=1.4, label="Overload penalty")
ax.plot(df_full["Omega_tau"], df_full["Normalized_score"], color=COLORS["net"], linewidth=2.0, label="Full-model score")

ax.axvline(BEST_OMEGA_TAU, color="#2d2d2d", linestyle=":", linewidth=1.0)
ax.set_xscale("log")
ax.set_xlabel(r"Dimensionless frequency, $\omega\tau_{\mathrm{ref}}$")
ax.set_ylabel("Normalized response")
ax.set_title(r"Frequency response with $\tau_{\mathrm{ref}}=\eta/E_2$", fontsize=9, loc="left")
ax.legend(fontsize=5.4, loc="lower left", ncol=2)

panel_label(ax, "A")
save_fig(fig, "part2_v3_panel_A_frequency_response_components.png")

# B. Finite full-model candidate window
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

ax.plot(df_full["Omega_tau"], df_full["Normalized_score"], color=COLORS["net"], linewidth=2.0, label="Full-model score")
ax.axhline(WINDOW_LEVEL, color="#777777", linestyle=":", linewidth=1.0, label="90% level")
ax.axvspan(WINDOW_LOW, WINDOW_HIGH, color=COLORS["window"], alpha=0.15, label="90% candidate window")
ax.axvline(BEST_OMEGA_TAU, color="#2d2d2d", linestyle=":", linewidth=1.0, label="Optimum")

ax.set_xscale("log")
ax.set_ylim(-0.03, 1.08)
ax.set_xlabel(r"Dimensionless frequency, $\omega\tau_{\mathrm{ref}}$")
ax.set_ylabel("Normalized score")
ax.set_title("Full-model candidate frequency window", fontsize=9, loc="left")
ax.legend(fontsize=5.6, loc="lower left")

panel_label(ax, "B")
save_fig(fig, "part2_v3_panel_B_candidate_frequency_window.png")

# C. Ablation
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

for mode in modes:
    sub = df_modes[df_modes["Mode"] == mode]
    ax.plot(
        sub["Omega_tau"],
        sub["Normalized_score"],
        color=mode_colors[mode],
        linewidth=1.6 if mode == "full" else 1.3,
        label=mode_labels[mode]
    )

ax.axvspan(WINDOW_LOW, WINDOW_HIGH, color="#A67B7B", alpha=0.08)
ax.set_xscale("log")
ax.set_xlabel(r"Dimensionless frequency, $\omega\tau_{\mathrm{ref}}$")
ax.set_ylabel("Normalized score")
ax.set_title("Ablation analysis of window formation", fontsize=9, loc="left")
ax.legend(fontsize=5.2, loc="upper right")

panel_label(ax, "C")
save_fig(fig, "part2_v3_panel_C_ablation_window_formation.png")

# D. Cellular cutoff sensitivity
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

for cutoff in CELLULAR_CUTOFF_SCENARIOS:
    sub = df_cutoff_curves[df_cutoff_curves["Cellular_cutoff_omega_tau"] == cutoff]
    ax.plot(sub["Omega_tau"], sub["Normalized_score"], linewidth=1.25, label=f"cutoff={cutoff:g}")

ax.axvspan(WINDOW_LOW, WINDOW_HIGH, color="#A67B7B", alpha=0.08)
ax.set_xscale("log")
ax.set_xlabel(r"Dimensionless frequency, $\omega\tau_{\mathrm{ref}}$")
ax.set_ylabel("Normalized score")
ax.set_title("Cellular-cutoff sensitivity", fontsize=9, loc="left")
ax.legend(fontsize=5.2, loc="upper right", ncol=2)

panel_label(ax, "D")
save_fig(fig, "part2_v3_panel_D_cellular_cutoff_sensitivity.png")

# E. Mean-load design space
pivot_design = (
    df_design
    .pivot_table(index="Mean_stress_fraction", columns="Omega_tau", values="Normalized_score", aggfunc="mean")
    .sort_index()
)

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

im = ax.imshow(
    pivot_design.values,
    origin="lower",
    aspect="auto",
    cmap=CMAP_SEQ,
    vmin=0,
    vmax=1,
    extent=[
        np.log10(pivot_design.columns.min()),
        np.log10(pivot_design.columns.max()),
        pivot_design.index.min(),
        pivot_design.index.max()
    ]
)

ax.axvspan(np.log10(WINDOW_LOW), np.log10(WINDOW_HIGH), color="#FFFFFF", alpha=0.20)
ax.set_xlabel(r"$\log_{10}(\omega\tau_{\mathrm{ref}})$")
ax.set_ylabel("Mean stress fraction")
ax.set_title("Mean-load design space", fontsize=9, loc="left")

cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
cbar.set_label("Normalized full-model score", fontsize=6.5)
cbar.ax.tick_params(labelsize=5.8)

panel_label(ax, "E")
save_fig(fig, "part2_v3_panel_E_mean_load_design_space.png")

# F. Monte Carlo best omega_tau
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

bins = np.logspace(np.log10(df_mc["Best_omega_tau"].min()), np.log10(df_mc["Best_omega_tau"].max()), 36)

ax.hist(
    df_mc["Best_omega_tau"],
    bins=bins,
    color="#7B9EA6",
    edgecolor="white",
    linewidth=0.4,
    alpha=0.9
)

ax.axvspan(WINDOW_LOW, WINDOW_HIGH, color="#A67B7B", alpha=0.15, label="Baseline 90% window")
ax.axvline(df_mc["Best_omega_tau"].median(), color="#2d2d2d", linestyle=":", linewidth=1.2, label="Median")

ax.set_xscale("log")
ax.set_xlabel(r"Best $\omega\tau_{\mathrm{ref}}$")
ax.set_ylabel("Monte Carlo count")
ax.set_title("Robustness of candidate window", fontsize=9, loc="left")
ax.legend(fontsize=5.7, loc="upper right")

panel_label(ax, "F")
save_fig(fig, "part2_v3_panel_F_monte_carlo_best_omega_tau.png")

# G. Absolute frequency scaling
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

ax.plot(
    df_scaling["Measured_or_assumed_tau_seconds"],
    df_scaling["Best_frequency_Hz"],
    color="#4A6572",
    linewidth=2.0,
    label="Best"
)

ax.fill_between(
    df_scaling["Measured_or_assumed_tau_seconds"],
    df_scaling["Window_low_frequency_Hz"],
    df_scaling["Window_high_frequency_Hz"],
    color="#A67B7B",
    alpha=0.15,
    label="90% window"
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Measured or assumed tissue/construct τ (s)")
ax.set_ylabel("Absolute frequency (Hz)")
ax.set_title("Absolute frequency requires measured τ", fontsize=9, loc="left")
ax.legend(fontsize=5.8, loc="upper right")

panel_label(ax, "G")
save_fig(fig, "part2_v3_panel_G_absolute_frequency_scaling.png")

# H. Phase, dissipation, high-frequency cost
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

ax.plot(df_full["Omega_tau"], df_full["Phase_norm"], color=COLORS["phase"], linewidth=1.6, label="Phase")
ax.plot(df_full["Omega_tau"], df_full["Dissipation_norm"], color=COLORS["dissipation"], linewidth=1.5, label="Dissipation")
ax.plot(df_full["Omega_tau"], df_full["High_frequency_cost"], color=COLORS["risk"], linewidth=1.5, label="High-frequency cost")

ax.axvspan(WINDOW_LOW, WINDOW_HIGH, color="#A67B7B", alpha=0.09)
ax.set_xscale("log")
ax.set_xlabel(r"Dimensionless frequency, $\omega\tau_{\mathrm{ref}}$")
ax.set_ylabel("Normalized dynamics")
ax.set_title("Phase, dissipation, and high-frequency dynamics", fontsize=9, loc="left")
ax.legend(fontsize=5.6, loc="upper right")

panel_label(ax, "H")
save_fig(fig, "part2_v3_panel_H_phase_dissipation_high_frequency.png")

# I. Sensitivity summary
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

ax.plot(
    df_cutoff_summary["Cellular_cutoff_omega_tau"],
    df_cutoff_summary["Best_omega_tau"],
    color="#4A6572",
    marker="o",
    linewidth=1.6,
    label="Cellular cutoff sensitivity"
)

ax.fill_between(
    df_cutoff_summary["Cellular_cutoff_omega_tau"],
    df_cutoff_summary["Window_low_omega_tau"],
    df_cutoff_summary["Window_high_omega_tau"],
    color="#A67B7B",
    alpha=0.15,
    label="90% window"
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Assumed cellular cutoff")
ax.set_ylabel(r"Best $\omega\tau_{\mathrm{ref}}$")
ax.set_title("Sensitivity of optimum to cellular cutoff", fontsize=9, loc="left")
ax.legend(fontsize=5.8, loc="upper left")

panel_label(ax, "I")
save_fig(fig, "part2_v3_panel_I_cutoff_sensitivity_summary.png")

# J. Decision table
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)
ax.axis("off")

table_data = pd.DataFrame([
    ["tau reference", "eta / E2"],
    ["Best omega tau", f"{BEST_OMEGA_TAU:.3g}"],
    ["90% window", f"{WINDOW_LOW:.3g} - {WINDOW_HIGH:.3g}"],
    ["Cell cutoff", f"{REFERENCE_CELLULAR_CUTOFF:.3g}, sensitivity"],
    ["HF penalty", f"{REFERENCE_HIGH_FREQ_CENTER:.3g}, sensitivity"],
    ["MC median best", f"{mc_summary['Median_best_omega_tau'].iloc[0]:.3g}"],
    ["Interpretation", "model-derived hypothesis"],
], columns=["Item", "Value"])

table = ax.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    loc="center",
    cellLoc="center",
    colLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(6.1)
table.scale(1.0, 1.35)

for _, cell in table.get_celld().items():
    cell.set_edgecolor("#D0D0D0")
    cell.set_linewidth(0.4)

ax.set_title("Part 2 v3 decision summary", fontsize=9, loc="left")

panel_label(ax, "J")
save_fig(fig, "part2_v3_panel_J_decision_summary.png")

# ============================================================
# 12. Combined 600 dpi overview
# ============================================================

def create_combined_overview_600dpi():
    if not PIL_AVAILABLE:
        print("Pillow is not available. Combined overview was skipped. Install pillow if needed: pip install pillow")
        return

    panel_files = [
        "part2_v3_panel_A_frequency_response_components.png",
        "part2_v3_panel_B_candidate_frequency_window.png",
        "part2_v3_panel_C_ablation_window_formation.png",
        "part2_v3_panel_D_cellular_cutoff_sensitivity.png",
        "part2_v3_panel_E_mean_load_design_space.png",
        "part2_v3_panel_F_monte_carlo_best_omega_tau.png",
        "part2_v3_panel_G_absolute_frequency_scaling.png",
        "part2_v3_panel_H_phase_dissipation_high_frequency.png",
        "part2_v3_panel_I_cutoff_sensitivity_summary.png",
        "part2_v3_panel_J_decision_summary.png",
    ]

    panel_paths = []
    for fn in panel_files:
        path = os.path.join(OUTDIR, fn)
        if os.path.exists(path):
            panel_paths.append(path)
        else:
            print(f"Missing panel for combined overview: {fn}")

    if len(panel_paths) == 0:
        print("No panels were found. Combined overview was not generated.")
        return

    images = []
    for path in panel_paths:
        img = Image.open(path).convert("RGB")
        images.append((os.path.basename(path), img))

    upscale_factor = 2

    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS

    upscaled_images = []
    for name, img in images:
        new_size = (img.size[0] * upscale_factor, img.size[1] * upscale_factor)
        img_up = img.resize(new_size, resample_filter)
        upscaled_images.append((name, img_up))

    ncols = 2
    nrows = math.ceil(len(upscaled_images) / ncols)

    max_w = max(img.size[0] for _, img in upscaled_images)
    max_h = max(img.size[1] for _, img in upscaled_images)

    outer_pad = 80
    inner_pad = 50
    border_px = 2

    cell_w = max_w + border_px * 2
    cell_h = max_h + border_px * 2

    total_w = outer_pad * 2 + ncols * cell_w + (ncols - 1) * inner_pad
    total_h = outer_pad * 2 + nrows * cell_h + (nrows - 1) * inner_pad

    canvas = Image.new("RGB", (total_w, total_h), "white")

    for idx, (name, img) in enumerate(upscaled_images):
        r = idx // ncols
        c = idx % ncols

        x0 = outer_pad + c * (cell_w + inner_pad)
        y0 = outer_pad + r * (cell_h + inner_pad)

        panel_canvas = Image.new("RGB", (cell_w, cell_h), "white")
        ix = border_px + (max_w - img.size[0]) // 2
        iy = border_px + (max_h - img.size[1]) // 2
        panel_canvas.paste(img, (ix, iy))
        panel_canvas = ImageOps.expand(panel_canvas, border=border_px, fill="#D0D0D0")

        canvas.paste(panel_canvas, (x0, y0))

    out_path = os.path.join(OUTDIR, "part2_v3_combined_overview_600dpi.png")
    canvas.save(out_path, dpi=(600, 600))

    out_path_alias = os.path.join(OUTDIR, "part2_v3_combined_overview.png")
    canvas.save(out_path_alias, dpi=(600, 600))

    print(f"Combined 600 dpi overview saved to: {out_path}")
    print(f"Combined 600 dpi overview alias saved to: {out_path_alias}")
    print(f"Combined overview includes {len(panel_paths)} panels.")

create_combined_overview_600dpi()

# ============================================================
# 13. Summary report
# ============================================================

print("\n" + "=" * 90)
print("Part 2 frequency-window analysis completed.")
print("=" * 90)
print("Script filename recommendation:")
print("part2_frequency_window.py")
print("")
print("Output folder:")
print(OUTDIR)
print("")
print("Hard correction:")
print("tau_ref = eta / E2. No eta/E1 is used.")
print("")
print("Baseline full-model summary:")
print(baseline_summary.to_string(index=False))
print("")
print("Ablation summary:")
print(df_ablation_summary.to_string(index=False))
print("")
print("Cellular cutoff sensitivity summary:")
print(df_cutoff_summary.to_string(index=False))
print("")
print("High-frequency penalty sensitivity summary:")
print(df_hf_summary.to_string(index=False))
print("")
print("Monte Carlo summary:")
print(mc_summary.to_string(index=False))
print("")
print("Decision table:")
print(decision_table.to_string(index=False))
print("")
print("Generated PNG files:")
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith(".png"):
        print(f)
print("")
print("Generated CSV files:")
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith(".csv"):
        print(f)
print("=" * 90)
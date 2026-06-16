# ============================================================
# Part 3: Hard-window Multi-objective Waveform Screening
# Reads Part 2 v3 outputs
#
# Required previous step:
# Run part2_frequency_window.py first.
#
# Core logic:
# 1. Read the Part 2 v3 frequency-window factor.
# 2. Preserve global exploratory efficiency results.
# 3. Use a HARD Part 2 v3 90% window restriction for primary ranking.
# 4. Add mechanical risk surrogate and smoothness surrogate.
# 5. Report:
#    - global exploratory best
#    - window-restricted efficiency best
#    - balanced multi-objective best
#    - near-optimal waveform classes
#    - Pareto frontier
#    - Monte Carlo robustness
#
# Important interpretation:
# Global exploratory best may occur outside the Part 2 v3 window.
# It is NOT the primary recommendation.
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

OUTDIR = "part3_waveform_screening_outputs"
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

CLASS_ORDER = ["Constant", "Sine", "Triangle", "Asymmetric", "Square", "Pulse"]

CLASS_COLORS = {
    "Constant": "#D9E3E5",
    "Sine": "#7B9EA6",
    "Triangle": "#B5A07A",
    "Asymmetric": "#8E85A6",
    "Square": "#8A9E82",
    "Pulse": "#A67B7B",
}

CMAP_SEQ = mcolors.LinearSegmentedColormap.from_list(
    "morandi_seq",
    ["#EEF3F4", "#C8D8DF", "#7B9EA6", "#4A6572", "#A67B7B"]
)

CLASS_CMAP = mcolors.ListedColormap([CLASS_COLORS[x] for x in CLASS_ORDER])

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

OMEGA_TAU_GRID = np.logspace(-2, 3, 120)
MEAN_FRACTION_GRID = np.array([0.30, 0.45, 0.60, 0.75, 0.90])

N_PHASE = 2048
N_MC = 1000
OMEGA_TAU_MC = np.logspace(-1, 2.3, 70)

NEAR_OPTIMAL_THRESHOLD_95 = 0.95
NEAR_OPTIMAL_THRESHOLD_97 = 0.97

EPS = 1e-12

# ============================================================
# 2. Helper functions
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
    Part 2 v3 reference definition.
    """
    return eta / E2

def frequency_from_omega_tau(omega_tau, E1, E2, eta):
    tau_ref = tau_ref_from_params(E1, E2, eta)
    return np.asarray(omega_tau, dtype=float) / (2.0 * np.pi * tau_ref)

def complex_compliance_sls(freq_hz, E1, E2, eta):
    """
    Standard linear solid frequency-domain compliance.

    J*(omega) = (1 + i omega eta / E2)
                /
                (E1 + i omega eta * (1 + E1/E2))
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

def short_class_name(x):
    if x == "Asymmetric":
        return "Asym"
    if x == "Constant":
        return "Const"
    if x == "Triangle":
        return "Tri"
    return str(x)

def require_columns(df, required_cols, source_name):
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"{source_name} is missing required columns: {', '.join(missing)}")

# ============================================================
# 3. Read Part 2 v3 outputs
# ============================================================

def locate_part2_v3_window_file():
    path = os.path.join(
        "part2_frequency_window_outputs",
        "part2_v3_frequency_response_all_ablation_modes.csv"
    )
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        "Cannot find Part 2 v3 window file. "
        "Run part2_frequency_window.py first. "
        "Expected: part2_frequency_window_outputs/"
        "part2_v3_frequency_response_all_ablation_modes.csv"
    )

def locate_part2_v3_summary_file():
    path = os.path.join(
        "part2_frequency_window_outputs",
        "part2_v3_baseline_full_model_window_summary.csv"
    )
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        "Cannot find Part 2 v3 summary file. "
        "Run part2_frequency_window.py first. "
        "Expected: part2_frequency_window_outputs/"
        "part2_v3_baseline_full_model_window_summary.csv"
    )

PART2_WINDOW_FILE = locate_part2_v3_window_file()
PART2_SUMMARY_FILE = locate_part2_v3_summary_file()

df_part2_all = pd.read_csv(PART2_WINDOW_FILE)
df_part2_summary = pd.read_csv(PART2_SUMMARY_FILE)

require_columns(
    df_part2_all,
    ["Omega_tau", "Frequency_Hz", "Normalized_score", "Mode"],
    "Part 2 v3 window file"
)

require_columns(
    df_part2_summary,
    [
        "Best_omega_tau",
        "Window_low_omega_tau",
        "Window_high_omega_tau",
        "Tau_ref_definition",
        "Tau_ref_eta_over_E2_seconds"
    ],
    "Part 2 v3 summary file"
)

df_part2_full = df_part2_all[df_part2_all["Mode"] == "full"].copy()

if len(df_part2_full) == 0:
    raise ValueError("Part 2 v3 file does not contain Mode == 'full'.")

df_part2_full = df_part2_full.sort_values("Omega_tau").reset_index(drop=True)

PART2_BEST_OMEGA_TAU = float(df_part2_summary["Best_omega_tau"].iloc[0])
PART2_WINDOW_LOW = float(df_part2_summary["Window_low_omega_tau"].iloc[0])
PART2_WINDOW_HIGH = float(df_part2_summary["Window_high_omega_tau"].iloc[0])
PART2_TAU_REF = float(df_part2_summary["Tau_ref_eta_over_E2_seconds"].iloc[0])

if str(df_part2_summary["Tau_ref_definition"].iloc[0]).strip() != "eta / E2":
    raise ValueError("Part 2 v3 summary does not report tau_ref = eta / E2.")

if abs(PART2_TAU_REF - tau_ref_from_params(BASE_PARAMS["E1"], BASE_PARAMS["E2"], BASE_PARAMS["eta"])) > 1e-8:
    raise ValueError("Part 2 v3 tau_ref does not match Part 3 baseline tau_ref.")

def interpolate_part2_window_factor(omega_tau_values):
    omega_tau_values = np.asarray(omega_tau_values, dtype=float)

    x = np.log10(df_part2_full["Omega_tau"].values)
    y = df_part2_full["Normalized_score"].values

    return np.interp(
        np.log10(omega_tau_values),
        x,
        y,
        left=y[0],
        right=y[-1]
    )

df_part2_full.to_csv(
    os.path.join(OUTDIR, "part3_v4_imported_part2_v3_full_window_factor.csv"),
    index=False
)

df_part2_summary.to_csv(
    os.path.join(OUTDIR, "part3_v4_imported_part2_v3_summary.csv"),
    index=False
)

# ============================================================
# 4. Waveform candidate definitions
# ============================================================

def generate_raw_waveform(phase, waveform_class, parameter=np.nan):
    phase = np.mod(np.asarray(phase, dtype=float), 1.0)

    if waveform_class == "Constant":
        return np.zeros_like(phase)

    if waveform_class == "Sine":
        return np.sin(2.0 * np.pi * phase)

    if waveform_class == "Triangle":
        return 4.0 * np.abs(phase - 0.5) - 1.0

    if waveform_class == "Asymmetric":
        rise_fraction = float(parameter)
        raw = np.where(
            phase < rise_fraction,
            phase / rise_fraction,
            1.0 - (phase - rise_fraction) / (1.0 - rise_fraction)
        )
        return 2.0 * raw - 1.0

    if waveform_class == "Square":
        return np.where(phase < 0.5, 1.0, -1.0)

    if waveform_class == "Pulse":
        duty = float(parameter)
        return np.where(phase < duty, 1.0, 0.0)

    raise ValueError(f"Unknown waveform class: {waveform_class}")

def standardize_dynamic_component(y):
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    rms = np.sqrt(np.mean(y ** 2))
    if rms < EPS:
        return np.zeros_like(y)
    return y / rms

def build_candidate_table():
    records = []

    records.append({
        "Design_ID": "Constant",
        "Waveform_class": "Constant",
        "Parameter_name": "None",
        "Parameter_value": np.nan,
        "Display_name": "Constant"
    })

    records.append({
        "Design_ID": "Sine",
        "Waveform_class": "Sine",
        "Parameter_name": "None",
        "Parameter_value": np.nan,
        "Display_name": "Sine"
    })

    records.append({
        "Design_ID": "Triangle",
        "Waveform_class": "Triangle",
        "Parameter_name": "None",
        "Parameter_value": np.nan,
        "Display_name": "Triangle"
    })

    for rise in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
        records.append({
            "Design_ID": f"Asymmetric_rise_{rise:.2f}",
            "Waveform_class": "Asymmetric",
            "Parameter_name": "Rise fraction",
            "Parameter_value": rise,
            "Display_name": f"Asym {rise:.2f}"
        })

    records.append({
        "Design_ID": "Square",
        "Waveform_class": "Square",
        "Parameter_name": "None",
        "Parameter_value": np.nan,
        "Display_name": "Square"
    })

    for duty in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        records.append({
            "Design_ID": f"Pulse_duty_{duty:.2f}",
            "Waveform_class": "Pulse",
            "Parameter_name": "Duty cycle",
            "Parameter_value": duty,
            "Display_name": f"Pulse {duty:.2f}"
        })

    return pd.DataFrame(records)

CANDIDATES = build_candidate_table()

CANDIDATES.to_csv(
    os.path.join(OUTDIR, "part3_v4_waveform_candidate_definitions.csv"),
    index=False
)

# ============================================================
# 5. Equal-RMS FFT-based SLS waveform evaluation
# ============================================================

def evaluate_periodic_design(
    waveform_class,
    parameter,
    omega_tau,
    mean_fraction,
    E1,
    E2,
    eta,
    target_rms_stress=TARGET_RMS_STRESS,
    n_phase=N_PHASE
):
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)

    raw = generate_raw_waveform(phase, waveform_class, parameter)
    dynamic = standardize_dynamic_component(raw)

    if waveform_class == "Constant":
        stress = target_rms_stress * np.ones_like(phase)
    else:
        mean_stress = target_rms_stress * mean_fraction
        dynamic_amp = np.sqrt(max(target_rms_stress ** 2 - mean_stress ** 2, EPS))
        stress = mean_stress + dynamic_amp * dynamic

    base_frequency_hz = float(frequency_from_omega_tau(omega_tau, E1, E2, eta))

    stress_fft = np.fft.rfft(stress)
    harmonic_index = np.arange(len(stress_fft))

    harmonic_frequency_hz = harmonic_index * base_frequency_hz
    harmonic_omega_tau = harmonic_index * omega_tau

    J = complex_compliance_sls(harmonic_frequency_hz, E1, E2, eta)

    strain_fft = stress_fft * J

    rate_fft = strain_fft.copy()
    rate_fft[0] = 0.0 + 0.0j
    rate_fft[1:] = strain_fft[1:] * (
        1j * 2.0 * np.pi * harmonic_frequency_hz[1:]
    )

    strain = np.fft.irfft(strain_fft, n=n_phase)
    strain_rate = np.fft.irfft(rate_fft, n=n_phase)

    abs_rate = np.abs(strain_rate)

    mean_abs_strain_rate = float(np.mean(abs_rate))
    integrated_strain_rate_per_cycle = float(np.mean(abs_rate))
    rms_strain_rate = float(np.sqrt(np.mean(strain_rate ** 2)))

    peak_abs_strain = float(np.max(np.abs(strain)))
    strain_range = float(np.max(strain) - np.min(strain))
    mean_strain = float(np.mean(strain))

    dissipative_work_proxy = float(np.mean(np.abs(stress * strain_rate)))

    rate_power = np.abs(rate_fft) ** 2
    rate_power[0] = 0.0
    total_rate_power = float(np.sum(rate_power) + EPS)

    harmonic_complexity = float(
        np.sum(rate_power[harmonic_index >= 2]) / total_rate_power
    )

    high_frequency_power_fraction = float(
        np.sum(rate_power[harmonic_omega_tau > PART2_WINDOW_HIGH]) / total_rate_power
    )

    very_high_frequency_power_fraction = float(
        np.sum(rate_power[harmonic_omega_tau > 10.0 * PART2_WINDOW_HIGH]) / total_rate_power
    )

    persistence_threshold = 0.20 * np.max(abs_rate + EPS)
    strain_rate_persistence = float(np.mean(abs_rate >= persistence_threshold))

    dominant_harmonic = int(harmonic_index[np.argmax(rate_power)]) if total_rate_power > EPS else 0

    isi_raw = integrated_strain_rate_per_cycle

    isi_peak_strain_ratio = isi_raw / (peak_abs_strain + EPS)
    isi_strain_range_ratio = isi_raw / (strain_range + EPS)

    return {
        "Base_frequency_Hz": base_frequency_hz,
        "Stress_RMS": float(np.sqrt(np.mean(stress ** 2))),
        "Stress_mean": float(np.mean(stress)),
        "Stress_dynamic_RMS": float(np.sqrt(np.mean((stress - np.mean(stress)) ** 2))),
        "Mean_abs_strain_rate": mean_abs_strain_rate,
        "Integrated_strain_rate_per_cycle": integrated_strain_rate_per_cycle,
        "RMS_strain_rate": rms_strain_rate,
        "Peak_abs_strain": peak_abs_strain,
        "Strain_range": strain_range,
        "Mean_strain": mean_strain,
        "Dissipative_work_proxy": dissipative_work_proxy,
        "Harmonic_complexity": harmonic_complexity,
        "High_frequency_power_fraction": high_frequency_power_fraction,
        "Very_high_frequency_power_fraction": very_high_frequency_power_fraction,
        "Strain_rate_persistence": strain_rate_persistence,
        "Dominant_harmonic": dominant_harmonic,
        "ISI_raw": isi_raw,
        "ISI_peak_strain_ratio": isi_peak_strain_ratio,
        "ISI_strain_range_ratio": isi_strain_range_ratio,
    }

# ============================================================
# 6. Full design-space simulation
# ============================================================

print("Running Part 3 hard-window multi-objective waveform screening...")

records = []

for _, cand in CANDIDATES.iterrows():
    for mean_fraction in MEAN_FRACTION_GRID:
        for omega_tau in OMEGA_TAU_GRID:
            metrics = evaluate_periodic_design(
                waveform_class=cand["Waveform_class"],
                parameter=cand["Parameter_value"],
                omega_tau=omega_tau,
                mean_fraction=mean_fraction,
                E1=BASE_PARAMS["E1"],
                E2=BASE_PARAMS["E2"],
                eta=BASE_PARAMS["eta"],
                n_phase=N_PHASE
            )

            records.append({
                "Design_ID": cand["Design_ID"],
                "Waveform_class": cand["Waveform_class"],
                "Parameter_name": cand["Parameter_name"],
                "Parameter_value": cand["Parameter_value"],
                "Display_name": cand["Display_name"],
                "Mean_stress_fraction": mean_fraction,
                "Omega_tau": omega_tau,
                **metrics
            })

df_design = pd.DataFrame(records)

df_design["Part2_v3_window_factor"] = interpolate_part2_window_factor(
    df_design["Omega_tau"].values
)

df_design["In_part2_v3_90pct_window"] = (
    (df_design["Omega_tau"] >= PART2_WINDOW_LOW) &
    (df_design["Omega_tau"] <= PART2_WINDOW_HIGH)
)

# ============================================================
# 7. Scoring
# ============================================================

df_design["Global_exploratory_ISI_peak_ratio"] = (
    df_design["ISI_peak_strain_ratio"] *
    df_design["Part2_v3_window_factor"]
)

df_design["Global_exploratory_ISI_range_ratio"] = (
    df_design["ISI_strain_range_ratio"] *
    df_design["Part2_v3_window_factor"]
)

df_design["Window_constrained_ISI_peak_ratio"] = np.where(
    df_design["In_part2_v3_90pct_window"],
    df_design["ISI_peak_strain_ratio"] * df_design["Part2_v3_window_factor"],
    0.0
)

df_design["Window_constrained_ISI_range_ratio"] = np.where(
    df_design["In_part2_v3_90pct_window"],
    df_design["ISI_strain_range_ratio"] * df_design["Part2_v3_window_factor"],
    0.0
)

df_design["ISI_raw_norm"] = minmax_normalize(df_design["ISI_raw"])
df_design["ISI_peak_strain_ratio_norm"] = minmax_normalize(df_design["ISI_peak_strain_ratio"])
df_design["ISI_strain_range_ratio_norm"] = minmax_normalize(df_design["ISI_strain_range_ratio"])

df_design["Global_exploratory_ISI_peak_ratio_norm"] = minmax_normalize(
    df_design["Global_exploratory_ISI_peak_ratio"]
)

df_design["Window_constrained_ISI_peak_ratio_norm"] = minmax_normalize(
    df_design["Window_constrained_ISI_peak_ratio"]
)

df_design["Window_constrained_ISI_range_ratio_norm"] = minmax_normalize(
    df_design["Window_constrained_ISI_range_ratio"]
)

df_design["Peak_abs_strain_norm"] = robust_normalize(df_design["Peak_abs_strain"])
df_design["Strain_range_norm"] = robust_normalize(df_design["Strain_range"])
df_design["Dissipative_work_norm"] = robust_normalize(df_design["Dissipative_work_proxy"])
df_design["Harmonic_complexity_norm"] = robust_normalize(df_design["Harmonic_complexity"])
df_design["High_frequency_power_fraction_norm"] = robust_normalize(df_design["High_frequency_power_fraction"])
df_design["Very_high_frequency_power_fraction_norm"] = robust_normalize(df_design["Very_high_frequency_power_fraction"])

df_design["Mechanical_risk_surrogate"] = (
    0.25 * df_design["Peak_abs_strain_norm"] +
    0.20 * df_design["Strain_range_norm"] +
    0.20 * df_design["Dissipative_work_norm"] +
    0.20 * df_design["High_frequency_power_fraction_norm"] +
    0.15 * df_design["Very_high_frequency_power_fraction_norm"]
)

df_design["Mechanical_risk_surrogate"] = np.clip(
    df_design["Mechanical_risk_surrogate"],
    0.0,
    1.0
)

df_design["Mechanical_safety_surrogate"] = 1.0 - df_design["Mechanical_risk_surrogate"]

df_design["Smoothness_penalty"] = (
    0.50 * df_design["Harmonic_complexity_norm"] +
    0.30 * df_design["High_frequency_power_fraction_norm"] +
    0.20 * df_design["Very_high_frequency_power_fraction_norm"]
)

df_design["Smoothness_penalty"] = np.clip(
    df_design["Smoothness_penalty"],
    0.0,
    1.0
)

df_design["Smoothness_score"] = 1.0 - df_design["Smoothness_penalty"]

df_design["Balanced_window_restricted_score"] = np.where(
    df_design["In_part2_v3_90pct_window"],
    df_design["Window_constrained_ISI_peak_ratio_norm"] *
    df_design["Mechanical_safety_surrogate"] *
    df_design["Smoothness_score"],
    0.0
)

df_design["Balanced_window_restricted_score_norm"] = minmax_normalize(
    df_design["Balanced_window_restricted_score"]
)

df_design.to_csv(
    os.path.join(OUTDIR, "part3_v4_full_multiobjective_design_space.csv"),
    index=False
)

df_primary = df_design[
    np.isclose(df_design["Mean_stress_fraction"], PRIMARY_MEAN_STRESS_FRACTION)
].copy()

# ============================================================
# 8. Best designs and summaries
# ============================================================

global_best_idx = int(df_primary["Global_exploratory_ISI_peak_ratio"].idxmax())
window_best_idx = int(df_primary["Window_constrained_ISI_peak_ratio"].idxmax())
balanced_best_idx = int(df_primary["Balanced_window_restricted_score"].idxmax())

df_global_best_primary = df_primary.loc[[global_best_idx]].copy()
df_window_best_primary = df_primary.loc[[window_best_idx]].copy()
df_balanced_best_primary = df_primary.loc[[balanced_best_idx]].copy()

df_global_best_primary.to_csv(
    os.path.join(OUTDIR, "part3_v4_global_exploratory_best_primary_design.csv"),
    index=False
)

df_window_best_primary.to_csv(
    os.path.join(OUTDIR, "part3_v4_window_restricted_efficiency_best_primary_design.csv"),
    index=False
)

df_balanced_best_primary.to_csv(
    os.path.join(OUTDIR, "part3_v4_balanced_best_primary_design.csv"),
    index=False
)

df_balanced_best_primary.to_csv(
    os.path.join(OUTDIR, "part3_v4_best_primary_design.csv"),
    index=False
)

condition_best = (
    df_primary
    .groupby(["Mean_stress_fraction", "Omega_tau"], as_index=False)["Window_constrained_ISI_peak_ratio"]
    .max()
    .rename(columns={"Window_constrained_ISI_peak_ratio": "Condition_best_ISI_peak_ratio"})
)

df_primary_near = df_primary.merge(
    condition_best,
    on=["Mean_stress_fraction", "Omega_tau"],
    how="left"
)

df_primary_near["Ratio_to_condition_best"] = (
    df_primary_near["Window_constrained_ISI_peak_ratio"] /
    np.maximum(df_primary_near["Condition_best_ISI_peak_ratio"], EPS)
)

df_primary_near["Near_optimal_95"] = (
    df_primary_near["Ratio_to_condition_best"] >= NEAR_OPTIMAL_THRESHOLD_95
)

df_primary_near["Near_optimal_97"] = (
    df_primary_near["Ratio_to_condition_best"] >= NEAR_OPTIMAL_THRESHOLD_97
)

df_primary_near.to_csv(
    os.path.join(OUTDIR, "part3_v4_primary_design_space_with_near_optimal_flags.csv"),
    index=False
)

class_summary_records = []

for cls in CLASS_ORDER:
    sub_all = df_design[df_design["Waveform_class"] == cls].copy()
    sub_primary = df_primary[df_primary["Waveform_class"] == cls].copy()
    sub_near = df_primary_near[df_primary_near["Waveform_class"] == cls].copy()

    if len(sub_all) == 0:
        continue

    global_best = sub_all.sort_values("Global_exploratory_ISI_peak_ratio", ascending=False).iloc[0]
    window_best = sub_all.sort_values("Window_constrained_ISI_peak_ratio", ascending=False).iloc[0]
    balanced_best = sub_all.sort_values("Balanced_window_restricted_score", ascending=False).iloc[0]

    class_summary_records.append({
        "Waveform_class": cls,

        "Global_best_design": global_best["Display_name"],
        "Global_best_omega_tau": float(global_best["Omega_tau"]),
        "Global_best_inside_part2_window": bool(global_best["In_part2_v3_90pct_window"]),

        "Window_best_design": window_best["Display_name"],
        "Window_best_omega_tau": float(window_best["Omega_tau"]),
        "Window_best_inside_part2_window": bool(window_best["In_part2_v3_90pct_window"]),

        "Balanced_best_design": balanced_best["Display_name"],
        "Balanced_best_omega_tau": float(balanced_best["Omega_tau"]),
        "Balanced_best_score_norm": float(balanced_best["Balanced_window_restricted_score_norm"]),

        "Median_ISI_raw": float(sub_all["ISI_raw"].median()),
        "Median_peak_abs_strain": float(sub_all["Peak_abs_strain"].median()),
        "Median_mechanical_risk_surrogate": float(sub_all["Mechanical_risk_surrogate"].median()),
        "Median_mechanical_safety_surrogate": float(sub_all["Mechanical_safety_surrogate"].median()),
        "Median_smoothness_score": float(sub_all["Smoothness_score"].median()),
        "Median_harmonic_complexity": float(sub_all["Harmonic_complexity"].median()),
        "Median_high_frequency_power_fraction": float(sub_all["High_frequency_power_fraction"].median()),

        "Primary_near_optimal_95_fraction": float(sub_near["Near_optimal_95"].mean()) if len(sub_near) > 0 else 0.0,
        "Primary_near_optimal_97_fraction": float(sub_near["Near_optimal_97"].mean()) if len(sub_near) > 0 else 0.0,
    })

df_class_summary = pd.DataFrame(class_summary_records)

df_class_summary.to_csv(
    os.path.join(OUTDIR, "part3_v4_waveform_class_summary.csv"),
    index=False
)

near_summary = (
    df_primary_near
    .groupby("Waveform_class", as_index=False)
    .agg(
        Near_optimal_95_count=("Near_optimal_95", "sum"),
        Near_optimal_97_count=("Near_optimal_97", "sum"),
        Total_primary_conditions=("Near_optimal_95", "count"),
        Median_ratio_to_best=("Ratio_to_condition_best", "median"),
        Max_ratio_to_best=("Ratio_to_condition_best", "max")
    )
)

near_summary["Near_optimal_95_fraction"] = (
    near_summary["Near_optimal_95_count"] /
    near_summary["Total_primary_conditions"]
)

near_summary["Near_optimal_97_fraction"] = (
    near_summary["Near_optimal_97_count"] /
    near_summary["Total_primary_conditions"]
)

near_summary.to_csv(
    os.path.join(OUTDIR, "part3_v4_near_optimal_waveform_summary_primary_mean.csv"),
    index=False
)

# ============================================================
# 9. Pareto and design recommendation matrices
# ============================================================

def compute_pareto_frontier(df_in, benefit_col, risk_col):
    df_tmp = df_in.copy()
    df_tmp = df_tmp.sort_values([risk_col, benefit_col], ascending=[True, False]).reset_index(drop=True)

    best_benefit = -np.inf
    keep = []

    for i, row in df_tmp.iterrows():
        benefit = row[benefit_col]
        if benefit > best_benefit + EPS:
            keep.append(i)
            best_benefit = benefit

    return df_tmp.loc[keep].copy()

df_pareto = compute_pareto_frontier(
    df_primary[df_primary["In_part2_v3_90pct_window"]].copy(),
    benefit_col="Window_constrained_ISI_peak_ratio",
    risk_col="Mechanical_risk_surrogate"
)

df_pareto.to_csv(
    os.path.join(OUTDIR, "part3_v4_pareto_window_efficiency_vs_mechanical_risk.csv"),
    index=False
)

best_nonsine_map = (
    df_design[
        (df_design["Waveform_class"] != "Sine") &
        (df_design["In_part2_v3_90pct_window"])
    ]
    .sort_values("Balanced_window_restricted_score", ascending=False)
    .groupby(["Mean_stress_fraction", "Omega_tau"])
    .head(1)
    .reset_index(drop=True)
)

class_to_num = {cls: i for i, cls in enumerate(CLASS_ORDER)}

matrix_best_nonsine = np.full((len(MEAN_FRACTION_GRID), len(OMEGA_TAU_GRID)), np.nan)

for i, mean_fraction in enumerate(MEAN_FRACTION_GRID):
    for j, omega_tau in enumerate(OMEGA_TAU_GRID):
        row = best_nonsine_map[
            (np.isclose(best_nonsine_map["Mean_stress_fraction"], mean_fraction)) &
            (np.isclose(best_nonsine_map["Omega_tau"], omega_tau))
        ]

        if len(row) > 0:
            matrix_best_nonsine[i, j] = class_to_num[row["Waveform_class"].iloc[0]]

freq_bins = [
    ("Below window", OMEGA_TAU_GRID[OMEGA_TAU_GRID < PART2_WINDOW_LOW]),
    ("Within window", OMEGA_TAU_GRID[(OMEGA_TAU_GRID >= PART2_WINDOW_LOW) & (OMEGA_TAU_GRID <= PART2_WINDOW_HIGH)]),
    ("Above window", OMEGA_TAU_GRID[OMEGA_TAU_GRID > PART2_WINDOW_HIGH]),
]

mean_bins = [
    ("Low mean", MEAN_FRACTION_GRID[MEAN_FRACTION_GRID <= 0.45]),
    ("Primary mean", MEAN_FRACTION_GRID[np.isclose(MEAN_FRACTION_GRID, PRIMARY_MEAN_STRESS_FRACTION)]),
    ("High mean", MEAN_FRACTION_GRID[MEAN_FRACTION_GRID >= 0.75]),
]

recommend_records = []

for freq_label, freq_values in freq_bins:
    for mean_label, mean_values in mean_bins:
        sub = df_design[
            df_design["Omega_tau"].isin(freq_values) &
            df_design["Mean_stress_fraction"].isin(mean_values)
        ].copy()

        if len(sub) == 0:
            continue

        best_global = sub.sort_values("Global_exploratory_ISI_peak_ratio", ascending=False).iloc[0]
        best_window = sub.sort_values("Window_constrained_ISI_peak_ratio", ascending=False).iloc[0]
        best_balanced = sub.sort_values("Balanced_window_restricted_score", ascending=False).iloc[0]

        nonsine_sub = sub[
            (sub["Waveform_class"] != "Sine") &
            (sub["In_part2_v3_90pct_window"])
        ].copy()

        if len(nonsine_sub) > 0:
            best_nonsine = nonsine_sub.sort_values("Balanced_window_restricted_score", ascending=False).iloc[0]
            nonsine_ratio = best_nonsine["Balanced_window_restricted_score"] / max(
                best_balanced["Balanced_window_restricted_score"],
                EPS
            )
        else:
            best_nonsine = best_balanced
            nonsine_ratio = np.nan

        recommend_records.append({
            "Frequency_zone": freq_label,
            "Mean_load_zone": mean_label,

            "Global_best_class": best_global["Waveform_class"],
            "Global_best_design": best_global["Display_name"],
            "Global_best_omega_tau": best_global["Omega_tau"],
            "Global_best_inside_window": best_global["In_part2_v3_90pct_window"],

            "Window_efficiency_best_class": best_window["Waveform_class"],
            "Window_efficiency_best_design": best_window["Display_name"],
            "Window_efficiency_best_omega_tau": best_window["Omega_tau"],
            "Window_efficiency_best_inside_window": best_window["In_part2_v3_90pct_window"],

            "Balanced_best_class": best_balanced["Waveform_class"],
            "Balanced_best_design": best_balanced["Display_name"],
            "Balanced_best_omega_tau": best_balanced["Omega_tau"],
            "Balanced_best_inside_window": best_balanced["In_part2_v3_90pct_window"],
            "Balanced_best_score_norm": best_balanced["Balanced_window_restricted_score_norm"],

            "Best_nonsine_balanced_class": best_nonsine["Waveform_class"],
            "Best_nonsine_balanced_design": best_nonsine["Display_name"],
            "Best_nonsine_balanced_omega_tau": best_nonsine["Omega_tau"],
            "Best_nonsine_balanced_ratio_to_balanced_best": nonsine_ratio,
        })

df_recommend = pd.DataFrame(recommend_records)

df_recommend.to_csv(
    os.path.join(OUTDIR, "part3_v4_design_recommendation_matrix.csv"),
    index=False
)

# ============================================================
# 10. Monte Carlo robustness
# ============================================================

print("Running Part 3 v4 Monte Carlo robustness...")

mc_records = []
mc_near_records = []

for k in range(N_MC):
    E1_mc = 10 ** np.random.uniform(np.log10(0.05), np.log10(5.0))
    E2_mc = E1_mc * 10 ** np.random.uniform(np.log10(2.0), np.log10(10.0))
    eta_mc = 10 ** np.random.uniform(np.log10(1.0), np.log10(200.0))

    mean_mc = PRIMARY_MEAN_STRESS_FRACTION

    temp_records = []

    for _, cand in CANDIDATES.iterrows():
        for omega_tau in OMEGA_TAU_MC:
            metrics = evaluate_periodic_design(
                waveform_class=cand["Waveform_class"],
                parameter=cand["Parameter_value"],
                omega_tau=omega_tau,
                mean_fraction=mean_mc,
                E1=E1_mc,
                E2=E2_mc,
                eta=eta_mc,
                n_phase=1024
            )

            temp_records.append({
                "Design_ID": cand["Design_ID"],
                "Waveform_class": cand["Waveform_class"],
                "Parameter_value": cand["Parameter_value"],
                "Display_name": cand["Display_name"],
                "Omega_tau": omega_tau,
                **metrics
            })

    df_tmp = pd.DataFrame(temp_records)

    df_tmp["Part2_v3_window_factor"] = interpolate_part2_window_factor(
        df_tmp["Omega_tau"].values
    )

    df_tmp["In_part2_v3_90pct_window"] = (
        (df_tmp["Omega_tau"] >= PART2_WINDOW_LOW) &
        (df_tmp["Omega_tau"] <= PART2_WINDOW_HIGH)
    )

    df_tmp["Global_exploratory_ISI_peak_ratio"] = (
        df_tmp["ISI_peak_strain_ratio"] *
        df_tmp["Part2_v3_window_factor"]
    )

    df_tmp["Window_constrained_ISI_peak_ratio"] = np.where(
        df_tmp["In_part2_v3_90pct_window"],
        df_tmp["ISI_peak_strain_ratio"] * df_tmp["Part2_v3_window_factor"],
        0.0
    )

    df_tmp["Window_constrained_ISI_peak_ratio_norm"] = minmax_normalize(
        df_tmp["Window_constrained_ISI_peak_ratio"]
    )

    df_tmp["Peak_abs_strain_norm"] = robust_normalize(df_tmp["Peak_abs_strain"])
    df_tmp["Strain_range_norm"] = robust_normalize(df_tmp["Strain_range"])
    df_tmp["Dissipative_work_norm"] = robust_normalize(df_tmp["Dissipative_work_proxy"])
    df_tmp["Harmonic_complexity_norm"] = robust_normalize(df_tmp["Harmonic_complexity"])
    df_tmp["High_frequency_power_fraction_norm"] = robust_normalize(df_tmp["High_frequency_power_fraction"])
    df_tmp["Very_high_frequency_power_fraction_norm"] = robust_normalize(df_tmp["Very_high_frequency_power_fraction"])

    df_tmp["Mechanical_risk_surrogate"] = (
        0.25 * df_tmp["Peak_abs_strain_norm"] +
        0.20 * df_tmp["Strain_range_norm"] +
        0.20 * df_tmp["Dissipative_work_norm"] +
        0.20 * df_tmp["High_frequency_power_fraction_norm"] +
        0.15 * df_tmp["Very_high_frequency_power_fraction_norm"]
    )

    df_tmp["Mechanical_risk_surrogate"] = np.clip(df_tmp["Mechanical_risk_surrogate"], 0.0, 1.0)
    df_tmp["Mechanical_safety_surrogate"] = 1.0 - df_tmp["Mechanical_risk_surrogate"]

    df_tmp["Smoothness_penalty"] = (
        0.50 * df_tmp["Harmonic_complexity_norm"] +
        0.30 * df_tmp["High_frequency_power_fraction_norm"] +
        0.20 * df_tmp["Very_high_frequency_power_fraction_norm"]
    )

    df_tmp["Smoothness_penalty"] = np.clip(df_tmp["Smoothness_penalty"], 0.0, 1.0)
    df_tmp["Smoothness_score"] = 1.0 - df_tmp["Smoothness_penalty"]

    df_tmp["Balanced_window_restricted_score"] = np.where(
        df_tmp["In_part2_v3_90pct_window"],
        df_tmp["Window_constrained_ISI_peak_ratio_norm"] *
        df_tmp["Mechanical_safety_surrogate"] *
        df_tmp["Smoothness_score"],
        0.0
    )

    global_best = df_tmp.loc[int(df_tmp["Global_exploratory_ISI_peak_ratio"].idxmax())]
    window_best = df_tmp.loc[int(df_tmp["Window_constrained_ISI_peak_ratio"].idxmax())]
    balanced_best = df_tmp.loc[int(df_tmp["Balanced_window_restricted_score"].idxmax())]

    class_best = (
        df_tmp
        .sort_values("Window_constrained_ISI_peak_ratio", ascending=False)
        .groupby("Waveform_class")
        .head(1)
        .reset_index(drop=True)
    )

    best_score = float(window_best["Window_constrained_ISI_peak_ratio"])

    class_best["Class_best_ratio_to_window_best"] = (
        class_best["Window_constrained_ISI_peak_ratio"] /
        max(best_score, EPS)
    )

    for _, row in class_best.iterrows():
        mc_near_records.append({
            "Iteration": k + 1,
            "Waveform_class": row["Waveform_class"],
            "Class_best_design_ID": row["Design_ID"],
            "Class_best_parameter_value": row["Parameter_value"],
            "Class_best_omega_tau": row["Omega_tau"],
            "Class_best_ratio_to_window_best": row["Class_best_ratio_to_window_best"],
            "Class_near_optimal_95": bool(row["Class_best_ratio_to_window_best"] >= NEAR_OPTIMAL_THRESHOLD_95),
            "Class_near_optimal_97": bool(row["Class_best_ratio_to_window_best"] >= NEAR_OPTIMAL_THRESHOLD_97),
        })

    mc_records.append({
        "Iteration": k + 1,
        "E1": E1_mc,
        "E2": E2_mc,
        "eta": eta_mc,
        "Tau_ref_eta_over_E2_seconds": tau_ref_from_params(E1_mc, E2_mc, eta_mc),

        "Global_best_waveform_class": global_best["Waveform_class"],
        "Global_best_design_ID": global_best["Design_ID"],
        "Global_best_omega_tau": global_best["Omega_tau"],
        "Global_best_inside_part2_v3_window": bool(global_best["In_part2_v3_90pct_window"]),

        "Window_best_waveform_class": window_best["Waveform_class"],
        "Window_best_design_ID": window_best["Design_ID"],
        "Window_best_omega_tau": window_best["Omega_tau"],
        "Window_best_inside_part2_v3_window": bool(window_best["In_part2_v3_90pct_window"]),

        "Balanced_best_waveform_class": balanced_best["Waveform_class"],
        "Balanced_best_design_ID": balanced_best["Design_ID"],
        "Balanced_best_omega_tau": balanced_best["Omega_tau"],
        "Balanced_best_inside_part2_v3_window": bool(balanced_best["In_part2_v3_90pct_window"]),
    })

df_mc = pd.DataFrame(mc_records)
df_mc_near = pd.DataFrame(mc_near_records)

df_mc.to_csv(
    os.path.join(OUTDIR, "part3_v4_monte_carlo_waveform_robustness.csv"),
    index=False
)

df_mc_near.to_csv(
    os.path.join(OUTDIR, "part3_v4_monte_carlo_near_optimal_by_class.csv"),
    index=False
)

df_mc_global_summary = (
    df_mc
    .groupby("Global_best_waveform_class")
    .size()
    .reset_index(name="Global_best_count")
)
df_mc_global_summary["Global_best_fraction"] = df_mc_global_summary["Global_best_count"] / N_MC

df_mc_window_summary = (
    df_mc
    .groupby("Window_best_waveform_class")
    .size()
    .reset_index(name="Window_best_count")
)
df_mc_window_summary["Window_best_fraction"] = df_mc_window_summary["Window_best_count"] / N_MC

df_mc_balanced_summary = (
    df_mc
    .groupby("Balanced_best_waveform_class")
    .size()
    .reset_index(name="Balanced_best_count")
)
df_mc_balanced_summary["Balanced_best_fraction"] = df_mc_balanced_summary["Balanced_best_count"] / N_MC

df_mc_global_summary.to_csv(
    os.path.join(OUTDIR, "part3_v4_monte_carlo_global_best_waveform_summary.csv"),
    index=False
)

df_mc_window_summary.to_csv(
    os.path.join(OUTDIR, "part3_v4_monte_carlo_window_best_waveform_summary.csv"),
    index=False
)

df_mc_balanced_summary.to_csv(
    os.path.join(OUTDIR, "part3_v4_monte_carlo_balanced_best_waveform_summary.csv"),
    index=False
)

df_mc_near_summary = (
    df_mc_near
    .groupby("Waveform_class", as_index=False)
    .agg(
        Near_optimal_95_count=("Class_near_optimal_95", "sum"),
        Near_optimal_97_count=("Class_near_optimal_97", "sum"),
        Median_ratio_to_window_best=("Class_best_ratio_to_window_best", "median"),
        Q1_ratio_to_window_best=("Class_best_ratio_to_window_best", lambda x: np.quantile(x, 0.25)),
        Q3_ratio_to_window_best=("Class_best_ratio_to_window_best", lambda x: np.quantile(x, 0.75)),
        Max_ratio_to_window_best=("Class_best_ratio_to_window_best", "max")
    )
)

df_mc_near_summary["Near_optimal_95_fraction"] = (
    df_mc_near_summary["Near_optimal_95_count"] / N_MC
)

df_mc_near_summary["Near_optimal_97_fraction"] = (
    df_mc_near_summary["Near_optimal_97_count"] / N_MC
)

df_mc_near_summary.to_csv(
    os.path.join(OUTDIR, "part3_v4_monte_carlo_near_optimal_summary.csv"),
    index=False
)

# ============================================================
# 11. Figures A-J
# ============================================================

# A. Equal-RMS waveform candidates
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

phase_plot = np.linspace(0.0, 1.0, N_PHASE, endpoint=False)

representatives = [
    ("Sine", "Sine", np.nan),
    ("Triangle", "Triangle", np.nan),
    ("Asym 0.15", "Asymmetric", 0.15),
    ("Square", "Square", np.nan),
    ("Pulse 0.20", "Pulse", 0.20),
]

for label, cls, par in representatives:
    y = standardize_dynamic_component(generate_raw_waveform(phase_plot, cls, par))
    ax.plot(
        phase_plot,
        y,
        color=CLASS_COLORS[cls],
        linewidth=1.25,
        alpha=0.92,
        label=label
    )

ax.set_xlabel("Normalized cycle")
ax.set_ylabel("Standardized dynamic load")
ax.set_title("Equal-RMS waveform candidates", fontsize=9, loc="left")
ax.legend(fontsize=5.8, loc="upper right", ncol=2)
panel_label(ax, "A")
save_fig(fig, "part3_v4_panel_A_equal_RMS_waveform_candidates.png")

# B. Part 2 hard window
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

ax.plot(
    df_part2_full["Omega_tau"],
    df_part2_full["Normalized_score"],
    color="#4A6572",
    linewidth=2.0,
    label="Imported Part 2 v3 full-model factor"
)

ax.axvspan(
    PART2_WINDOW_LOW,
    PART2_WINDOW_HIGH,
    color="#A67B7B",
    alpha=0.15,
    label="Hard 90% window"
)

ax.axvline(
    float(df_global_best_primary["Omega_tau"].iloc[0]),
    color="#A67B7B",
    linestyle="--",
    linewidth=1.0,
    label="Global best"
)

ax.axvline(
    float(df_window_best_primary["Omega_tau"].iloc[0]),
    color="#2d2d2d",
    linestyle=":",
    linewidth=1.2,
    label="Window best"
)

ax.axvline(
    float(df_balanced_best_primary["Omega_tau"].iloc[0]),
    color="#8E85A6",
    linestyle="-.",
    linewidth=1.1,
    label="Balanced best"
)

ax.set_xscale("log")
ax.set_ylim(-0.03, 1.08)
ax.set_xlabel(r"Dimensionless frequency, $\omega\tau_{\mathrm{ref}}$")
ax.set_ylabel("Part 2 v3 factor")
ax.set_title("Hard Part 2 window constraint", fontsize=9, loc="left")
ax.legend(fontsize=5.1, loc="upper right")
panel_label(ax, "B")
save_fig(fig, "part3_v4_panel_B_hard_part2_window_constraint.png")

# C. Global vs window-restricted efficiency
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

for cls in CLASS_ORDER:
    sub = df_primary[df_primary["Waveform_class"] == cls].copy()
    curve = (
        sub
        .groupby("Omega_tau", as_index=False)
        .agg(
            Global=("Global_exploratory_ISI_peak_ratio_norm", "max"),
            Window=("Window_constrained_ISI_peak_ratio_norm", "max")
        )
    )

    ax.plot(
        curve["Omega_tau"],
        curve["Global"],
        color=CLASS_COLORS[cls],
        linewidth=0.9,
        alpha=0.35
    )

    ax.plot(
        curve["Omega_tau"],
        curve["Window"],
        color=CLASS_COLORS[cls],
        linewidth=1.5,
        alpha=0.95,
        label=cls
    )

ax.axvspan(PART2_WINDOW_LOW, PART2_WINDOW_HIGH, color="#A67B7B", alpha=0.08)
ax.set_xscale("log")
ax.set_xlabel(r"Dimensionless frequency, $\omega\tau_{\mathrm{ref}}$")
ax.set_ylabel("Normalized efficiency")
ax.set_title("Global exploratory versus hard-window efficiency", fontsize=9, loc="left")
ax.legend(fontsize=5.2, loc="upper right", ncol=2)
panel_label(ax, "C")
save_fig(fig, "part3_v4_panel_C_global_vs_window_efficiency.png")

# D. Top hard-window efficiency designs
df_top_window = (
    df_primary
    .sort_values("Window_constrained_ISI_peak_ratio", ascending=False)
    .head(10)
    .copy()
)

df_top_window["Short_label"] = (
    df_top_window["Display_name"] +
    "\n" +
    df_top_window["Omega_tau"].map(lambda x: f"ωτ={x:.1f}")
)

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

bars = ax.bar(
    np.arange(len(df_top_window)),
    df_top_window["Window_constrained_ISI_peak_ratio_norm"],
    color=[CLASS_COLORS[x] for x in df_top_window["Waveform_class"]],
    edgecolor="white",
    linewidth=0.6,
    width=0.68
)

ax.set_xticks(np.arange(len(df_top_window)))
ax.set_xticklabels(df_top_window["Short_label"], rotation=35, ha="right", fontsize=5.2)
ax.set_ylabel("Normalized hard-window ISI / peak-strain")
ax.set_title("Top hard-window efficiency designs", fontsize=9, loc="left")
ax.set_ylim(0, 1.12)

for bar, val in zip(bars, df_top_window["Window_constrained_ISI_peak_ratio_norm"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.025,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#2d2d2d"
    )

panel_label(ax, "D")
save_fig(fig, "part3_v4_panel_D_top_hard_window_efficiency_designs.png")

# E. Top balanced designs
df_top_balanced = (
    df_primary
    .sort_values("Balanced_window_restricted_score", ascending=False)
    .head(10)
    .copy()
)

df_top_balanced["Short_label"] = (
    df_top_balanced["Display_name"] +
    "\n" +
    df_top_balanced["Omega_tau"].map(lambda x: f"ωτ={x:.1f}")
)

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

bars = ax.bar(
    np.arange(len(df_top_balanced)),
    df_top_balanced["Balanced_window_restricted_score_norm"],
    color=[CLASS_COLORS[x] for x in df_top_balanced["Waveform_class"]],
    edgecolor="white",
    linewidth=0.6,
    width=0.68
)

ax.set_xticks(np.arange(len(df_top_balanced)))
ax.set_xticklabels(df_top_balanced["Short_label"], rotation=35, ha="right", fontsize=5.2)
ax.set_ylabel("Normalized balanced score")
ax.set_title("Top balanced hard-window designs", fontsize=9, loc="left")
ax.set_ylim(0, 1.12)

for bar, val in zip(bars, df_top_balanced["Balanced_window_restricted_score_norm"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.025,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#2d2d2d"
    )

panel_label(ax, "E")
save_fig(fig, "part3_v4_panel_E_top_balanced_designs.png")

# F. Efficiency-risk-smoothness design space
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

for cls in CLASS_ORDER:
    sub = df_primary[
        (df_primary["Waveform_class"] == cls) &
        (df_primary["In_part2_v3_90pct_window"])
    ].copy()

    if len(sub) == 0:
        continue

    ax.scatter(
        sub["Mechanical_risk_surrogate"],
        sub["Window_constrained_ISI_peak_ratio_norm"],
        s=12 + 25 * sub["Smoothness_score"],
        color=CLASS_COLORS[cls],
        alpha=0.45 if cls not in ["Sine", "Triangle", "Asymmetric"] else 0.65,
        edgecolor="none",
        label=cls
    )

ax.set_xlabel("Mechanical risk surrogate")
ax.set_ylabel("Window-restricted efficiency")
ax.set_title("Efficiency-risk-smoothness design space", fontsize=9, loc="left")
ax.legend(fontsize=5.2, loc="lower right", ncol=2)
panel_label(ax, "F")
save_fig(fig, "part3_v4_panel_F_efficiency_risk_smoothness_space.png")

# G. Best non-sine balanced map
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

im = ax.imshow(
    matrix_best_nonsine,
    origin="lower",
    aspect="auto",
    cmap=CLASS_CMAP,
    vmin=-0.5,
    vmax=len(CLASS_ORDER) - 0.5,
    extent=[
        np.log10(OMEGA_TAU_GRID.min()),
        np.log10(OMEGA_TAU_GRID.max()),
        MEAN_FRACTION_GRID.min(),
        MEAN_FRACTION_GRID.max()
    ]
)

ax.axvspan(np.log10(PART2_WINDOW_LOW), np.log10(PART2_WINDOW_HIGH), color="#FFFFFF", alpha=0.20)
ax.set_xlabel(r"$\log_{10}(\omega\tau_{\mathrm{ref}})$")
ax.set_ylabel("Mean stress fraction")
ax.set_title("Best non-sine balanced alternative", fontsize=9, loc="left")

cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.02, ticks=np.arange(len(CLASS_ORDER)))
cbar.ax.set_yticklabels(CLASS_ORDER, fontsize=5.8)
cbar.ax.tick_params(labelsize=5.8)
panel_label(ax, "G")
save_fig(fig, "part3_v4_panel_G_best_nonsine_balanced_map.png")

# H. Pareto frontier
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

for cls in CLASS_ORDER:
    sub = df_primary[
        (df_primary["Waveform_class"] == cls) &
        (df_primary["In_part2_v3_90pct_window"])
    ].copy()

    ax.scatter(
        sub["Mechanical_risk_surrogate"],
        sub["Window_constrained_ISI_peak_ratio"],
        color=CLASS_COLORS[cls],
        s=14,
        alpha=0.42,
        edgecolor="none",
        label=cls
    )

ax.plot(
    df_pareto["Mechanical_risk_surrogate"],
    df_pareto["Window_constrained_ISI_peak_ratio"],
    color="#2d2d2d",
    linewidth=1.4,
    linestyle=":",
    label="Pareto frontier"
)

ax.set_xlabel("Mechanical risk surrogate")
ax.set_ylabel("Hard-window ISI / peak-strain")
ax.set_title("Pareto frontier under hard window", fontsize=9, loc="left")
ax.legend(fontsize=5.1, loc="lower right", ncol=2)
panel_label(ax, "H")
save_fig(fig, "part3_v4_panel_H_pareto_efficiency_vs_risk.png")

# I. Monte Carlo summaries
mc_window_plot = (
    df_mc_window_summary
    .set_index("Window_best_waveform_class")
    .reindex(CLASS_ORDER)
    .fillna(0)
    .reset_index()
    .rename(columns={"Window_best_waveform_class": "Waveform_class"})
)

mc_balanced_plot = (
    df_mc_balanced_summary
    .set_index("Balanced_best_waveform_class")
    .reindex(CLASS_ORDER)
    .fillna(0)
    .reset_index()
    .rename(columns={"Balanced_best_waveform_class": "Waveform_class"})
)

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

x = np.arange(len(CLASS_ORDER))
width = 0.34

ax.bar(
    x - width / 2,
    mc_window_plot["Window_best_fraction"],
    width=width,
    color="#7B9EA6",
    edgecolor="white",
    linewidth=0.5,
    label="Window efficiency best"
)

ax.bar(
    x + width / 2,
    mc_balanced_plot["Balanced_best_fraction"],
    width=width,
    color="#A67B7B",
    edgecolor="white",
    linewidth=0.5,
    label="Balanced best"
)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_ORDER, rotation=30, ha="right", fontsize=6.0)
ax.set_ylabel("Monte Carlo fraction")
ax.set_title("Window-efficiency versus balanced robustness", fontsize=9, loc="left")
ax.set_ylim(0, 1.12)
ax.legend(fontsize=5.7, loc="upper right")

panel_label(ax, "I")
save_fig(fig, "part3_v4_panel_I_monte_carlo_window_vs_balanced.png")

# J. Recommendation matrix
freq_order = ["Below window", "Within window", "Above window"]
mean_order = ["Low mean", "Primary mean", "High mean"]

matrix_recommend = np.full((len(mean_order), len(freq_order)), np.nan)

for i, mean_label in enumerate(mean_order):
    for j, freq_label in enumerate(freq_order):
        row = df_recommend[
            (df_recommend["Mean_load_zone"] == mean_label) &
            (df_recommend["Frequency_zone"] == freq_label)
        ]
        if len(row) > 0:
            cls = row["Balanced_best_class"].iloc[0]
            if cls in class_to_num:
                matrix_recommend[i, j] = class_to_num[cls]

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

im = ax.imshow(
    matrix_recommend,
    cmap=CLASS_CMAP,
    aspect="auto",
    vmin=-0.5,
    vmax=len(CLASS_ORDER) - 0.5
)

ax.set_xticks(np.arange(len(freq_order)))
ax.set_xticklabels(freq_order, fontsize=5.8, rotation=15, ha="right")
ax.set_yticks(np.arange(len(mean_order)))
ax.set_yticklabels(mean_order, fontsize=6.0)
ax.set_title("Global, window, and balanced recommendation", fontsize=9, loc="left")

for i, mean_label in enumerate(mean_order):
    for j, freq_label in enumerate(freq_order):
        row = df_recommend[
            (df_recommend["Mean_load_zone"] == mean_label) &
            (df_recommend["Frequency_zone"] == freq_label)
        ]
        if len(row) > 0:
            g = short_class_name(row["Global_best_class"].iloc[0])
            w = short_class_name(row["Window_efficiency_best_class"].iloc[0])
            b = short_class_name(row["Balanced_best_class"].iloc[0])
            text_label = f"G:{g}\nW:{w}\nB:{b}"
            ax.text(j, i, text_label, ha="center", va="center", fontsize=5.0, color="#2d2d2d")

cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.02, ticks=np.arange(len(CLASS_ORDER)))
cbar.ax.set_yticklabels(CLASS_ORDER, fontsize=5.8)
cbar.ax.tick_params(labelsize=5.8)

panel_label(ax, "J")
save_fig(fig, "part3_v4_panel_J_recommendation_matrix.png")

# ============================================================
# 12. Validation, interpretation_note, score support
# ============================================================

validation_table = pd.DataFrame([
    {
        "Criterion": "Part 2 dependency",
        "Value": PART2_WINDOW_FILE,
        "Interpretation": "Part 3 v4 reads the Part 2 v3 full-model window factor."
    },
    {
        "Criterion": "Hard-window correction",
        "Value": "Designs outside the Part 2 v3 90% window receive zero primary window-constrained score.",
        "Interpretation": "The primary Part 3 ranking now respects the Part 2 frequency window."
    },
    {
        "Criterion": "Global exploratory best",
        "Value": str(df_global_best_primary["Display_name"].iloc[0]),
        "Interpretation": "Highest exploratory mechanical efficiency; not the primary recommendation if outside the window."
    },
    {
        "Criterion": "Window-restricted efficiency best",
        "Value": str(df_window_best_primary["Display_name"].iloc[0]),
        "Interpretation": "Highest ISI/peak-strain ratio inside the Part 2 v3 hard window."
    },
    {
        "Criterion": "Balanced best",
        "Value": str(df_balanced_best_primary["Display_name"].iloc[0]),
        "Interpretation": "Best multi-objective candidate combining window-restricted efficiency, mechanical safety surrogate, and smoothness."
    },
    {
        "Criterion": "Window best inside Part 2 v3 window",
        "Value": bool(df_window_best_primary["In_part2_v3_90pct_window"].iloc[0]),
        "Interpretation": "Should be True."
    },
    {
        "Criterion": "Balanced best inside Part 2 v3 window",
        "Value": bool(df_balanced_best_primary["In_part2_v3_90pct_window"].iloc[0]),
        "Interpretation": "Should be True."
    },
])

validation_table.to_csv(
    os.path.join(OUTDIR, "part3_v4_validation_table.csv"),
    index=False
)

interpretation_note_table = pd.DataFrame([
    {
        "Rule": "Do not treat global exploratory best as the primary recommendation",
        "Meaning": "Global best can reveal aggressive high-efficiency candidates but may violate the Part 2 window."
    },
    {
        "Rule": "Primary ranking must respect the Part 2 v3 hard window",
        "Meaning": "Designs outside the Part 2 90% window cannot win the main Part 3 ranking."
    },
    {
        "Rule": "Balanced score is a surrogate",
        "Meaning": "It combines mechanical efficiency, smoothness, and risk surrogates, but is not biological proof."
    },
    {
        "Rule": "Sine is conservative only if supported by smoothness and risk metrics",
        "Meaning": "Do not call sine safest without pointing to low harmonic burden and low mechanical-risk surrogate."
    },
    {
        "Rule": "Constant is inactive reference",
        "Meaning": "Low stimulation is not equivalent to therapeutic optimality."
    },
])

interpretation_note_table.to_csv(
    os.path.join(OUTDIR, "part3_v4_interpretation_interpretation_notes.csv"),
    index=False
)


# ============================================================
# 13. Combined 600 dpi overview
# ============================================================

def create_combined_overview_600dpi():
    if not PIL_AVAILABLE:
        print("Pillow is not available. Combined overview was skipped. Install pillow if needed: pip install pillow")
        return

    panel_files = [
        "part3_v4_panel_A_equal_RMS_waveform_candidates.png",
        "part3_v4_panel_B_hard_part2_window_constraint.png",
        "part3_v4_panel_C_global_vs_window_efficiency.png",
        "part3_v4_panel_D_top_hard_window_efficiency_designs.png",
        "part3_v4_panel_E_top_balanced_designs.png",
        "part3_v4_panel_F_efficiency_risk_smoothness_space.png",
        "part3_v4_panel_G_best_nonsine_balanced_map.png",
        "part3_v4_panel_H_pareto_efficiency_vs_risk.png",
        "part3_v4_panel_I_monte_carlo_window_vs_balanced.png",
        "part3_v4_panel_J_recommendation_matrix.png",
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

    out_path = os.path.join(OUTDIR, "part3_v4_combined_overview_600dpi.png")
    canvas.save(out_path, dpi=(600, 600))

    out_path_alias = os.path.join(OUTDIR, "part3_v4_combined_overview.png")
    canvas.save(out_path_alias, dpi=(600, 600))

    print(f"Combined 600 dpi overview saved to: {out_path}")
    print(f"Combined 600 dpi overview alias saved to: {out_path_alias}")
    print(f"Combined overview includes {len(panel_paths)} panels.")

create_combined_overview_600dpi()

# ============================================================
# 14. Summary report
# ============================================================

print("\n" + "=" * 90)
print("Part 3 waveform-screening analysis completed.")
print("=" * 90)
print("Script filename recommendation:")
print("part3_waveform_screening.py")
print("")
print("Output folder:")
print(OUTDIR)
print("")
print("Required Part 2 v3 input:")
print(PART2_WINDOW_FILE)
print(PART2_SUMMARY_FILE)
print("")
print("Hard correction:")
print("Primary Part 3 ranking now uses a hard Part 2 v3 90% window restriction.")
print("")
print("Part 2 v3 summary used by Part 3:")
print(df_part2_summary.to_string(index=False))
print("")
print("Global exploratory best primary design:")
print(df_global_best_primary[[
    "Design_ID",
    "Waveform_class",
    "Display_name",
    "Parameter_name",
    "Parameter_value",
    "Mean_stress_fraction",
    "Omega_tau",
    "Base_frequency_Hz",
    "Part2_v3_window_factor",
    "In_part2_v3_90pct_window",
    "Global_exploratory_ISI_peak_ratio_norm",
]].to_string(index=False))
print("")
print("Window-restricted efficiency best primary design:")
print(df_window_best_primary[[
    "Design_ID",
    "Waveform_class",
    "Display_name",
    "Parameter_name",
    "Parameter_value",
    "Mean_stress_fraction",
    "Omega_tau",
    "Base_frequency_Hz",
    "Part2_v3_window_factor",
    "In_part2_v3_90pct_window",
    "Window_constrained_ISI_peak_ratio_norm",
]].to_string(index=False))
print("")
print("Balanced best primary design:")
print(df_balanced_best_primary[[
    "Design_ID",
    "Waveform_class",
    "Display_name",
    "Parameter_name",
    "Parameter_value",
    "Mean_stress_fraction",
    "Omega_tau",
    "Base_frequency_Hz",
    "Part2_v3_window_factor",
    "In_part2_v3_90pct_window",
    "Window_constrained_ISI_peak_ratio_norm",
    "Mechanical_risk_surrogate",
    "Mechanical_safety_surrogate",
    "Smoothness_score",
    "Balanced_window_restricted_score_norm",
]].to_string(index=False))
print("")
print("Waveform class summary:")
print(df_class_summary.to_string(index=False))
print("")
print("Near-optimal waveform summary:")
print(near_summary.to_string(index=False))
print("")
print("Monte Carlo global best summary:")
print(df_mc_global_summary.to_string(index=False))
print("")
print("Monte Carlo window-restricted best summary:")
print(df_mc_window_summary.to_string(index=False))
print("")
print("Monte Carlo balanced best summary:")
print(df_mc_balanced_summary.to_string(index=False))
print("")
print("Monte Carlo near-optimal summary:")
print(df_mc_near_summary.to_string(index=False))
print("")
print("Validation table:")
print(validation_table.to_string(index=False))
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
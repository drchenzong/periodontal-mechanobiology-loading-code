# ============================================================
# Part 4: Experimental Protocol Translation
# Reads Part 2 v3 and Part 3 v4 outputs
#
# Required previous steps:
# 1. Run part2_frequency_window.py
# 2. Run part3_waveform_screening.py
#
# Core logic:
# 1. Read manuscript-linked Part 2 v3 frequency-window summary.
# 2. Read manuscript-linked Part 3 v4 multi-objective waveform design space.
# 3. Use Balanced_window_restricted_score as the primary protocol score.
# 4. Preserve global exploratory and window-efficiency scores for interpretation.
# 5. Translate omega*tau into absolute frequency, period, cycle counts, and validation protocols.
#
# This script does not generate clinical treatment prescriptions.
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
# 0. Output folder and publication style
# ============================================================

OUTDIR = "part4_protocol_translation_outputs"
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

CMAP_RISK = mcolors.LinearSegmentedColormap.from_list(
    "morandi_risk",
    ["#EEF3F4", "#B5A07A", "#A67B7B"]
)

CLASS_CMAP = mcolors.ListedColormap([CLASS_COLORS[x] for x in CLASS_ORDER])

# ============================================================
# 1. Protocol-translation settings
# ============================================================

PRIMARY_MEAN_STRESS_FRACTION = 0.60
WINDOW_LEVEL = 0.90

# Conservative tau range for protocol translation.
# These values are scaling anchors, not measured universal PDL constants.
TAU_RANGE_SECONDS = np.logspace(0, 2, 220)
TAU_ANCHORS_SECONDS = np.array([1, 2, 3, 5, 10, 20, 30, 60, 100], dtype=float)

SESSION_DURATION_MINUTES = np.array([5, 10, 20, 30, 45, 60], dtype=float)
DAILY_SESSIONS = np.array([1, 2, 3], dtype=float)

VALIDATION_TAU_SECONDS = np.array([3, 10, 20, 60, 100], dtype=float)
VALIDATION_WAVEFORMS = ["Sine", "Triangle", "Asymmetric", "Square", "Pulse"]

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

def frequency_from_omega_tau(omega_tau, tau_seconds):
    return np.asarray(omega_tau, dtype=float) / (2.0 * np.pi * np.asarray(tau_seconds, dtype=float))

def period_from_omega_tau(omega_tau, tau_seconds):
    freq = frequency_from_omega_tau(omega_tau, tau_seconds)
    return 1.0 / np.maximum(freq, EPS)

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
# 3. Locate and read Part 2 v3 and Part 3 v4 outputs
# ============================================================

def locate_part2_v3_summary_file():
    path = os.path.join(
        "part2_frequency_window_outputs",
        "part2_v3_baseline_full_model_window_summary.csv"
    )
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        "Cannot find Part 2 v3 summary file. "
        "Run part2_frequency_window.py first."
    )

def locate_part2_v3_full_window_file():
    path = os.path.join(
        "part2_frequency_window_outputs",
        "part2_v3_frequency_response_all_ablation_modes.csv"
    )
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        "Cannot find Part 2 v3 full window file. "
        "Run part2_frequency_window.py first."
    )

def locate_part3_v4_design_space_file():
    path = os.path.join(
        "part3_waveform_screening_outputs",
        "part3_v4_full_multiobjective_design_space.csv"
    )
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        "Cannot find Part 3 v4 design-space file. "
        "Run part3_waveform_screening.py first. "
        "Expected: part3_waveform_screening_outputs/"
        "part3_v4_full_multiobjective_design_space.csv"
    )

def locate_part3_v4_best_file():
    path = os.path.join(
        "part3_waveform_screening_outputs",
        "part3_v4_balanced_best_primary_design.csv"
    )
    if os.path.exists(path):
        return path
    return None

PART2_SUMMARY_FILE = locate_part2_v3_summary_file()
PART2_WINDOW_FILE = locate_part2_v3_full_window_file()
PART3_DESIGN_FILE = locate_part3_v4_design_space_file()
PART3_BEST_FILE = locate_part3_v4_best_file()

df_part2_summary = pd.read_csv(PART2_SUMMARY_FILE)
df_part2_all = pd.read_csv(PART2_WINDOW_FILE)
df_part3 = pd.read_csv(PART3_DESIGN_FILE)

require_columns(
    df_part2_summary,
    [
        "Best_omega_tau",
        "Window_low_omega_tau",
        "Window_high_omega_tau",
        "Tau_ref_definition",
        "Tau_ref_eta_over_E2_seconds"
    ],
    "Part 2 v3 summary"
)

require_columns(
    df_part2_all,
    ["Omega_tau", "Normalized_score", "Mode"],
    "Part 2 v3 window file"
)

required_part3_cols = [
    "Design_ID",
    "Waveform_class",
    "Display_name",
    "Mean_stress_fraction",
    "Omega_tau",
    "Base_frequency_Hz",
    "Part2_v3_window_factor",
    "In_part2_v3_90pct_window",
    "ISI_raw",
    "Peak_abs_strain",
    "ISI_peak_strain_ratio",
    "Window_constrained_ISI_peak_ratio",
    "Window_constrained_ISI_peak_ratio_norm",
    "Global_exploratory_ISI_peak_ratio",
    "Global_exploratory_ISI_peak_ratio_norm",
    "Mechanical_risk_surrogate",
    "Mechanical_safety_surrogate",
    "Smoothness_score",
    "Balanced_window_restricted_score",
    "Balanced_window_restricted_score_norm",
    "Harmonic_complexity",
    "High_frequency_power_fraction",
    "Very_high_frequency_power_fraction",
]

require_columns(df_part3, required_part3_cols, "Part 3 v4 design space")

if str(df_part2_summary["Tau_ref_definition"].iloc[0]).strip() != "eta / E2":
    raise ValueError("Part 2 v3 summary does not report tau_ref = eta / E2.")

BEST_OMEGA_TAU = float(df_part2_summary["Best_omega_tau"].iloc[0])
WINDOW_LOW = float(df_part2_summary["Window_low_omega_tau"].iloc[0])
WINDOW_HIGH = float(df_part2_summary["Window_high_omega_tau"].iloc[0])
PART2_TAU_REF = float(df_part2_summary["Tau_ref_eta_over_E2_seconds"].iloc[0])

df_part2_full = df_part2_all[df_part2_all["Mode"] == "full"].copy()
df_part2_full = df_part2_full.sort_values("Omega_tau").reset_index(drop=True)

if len(df_part2_full) == 0:
    raise ValueError("Part 2 v3 full window file has no Mode == 'full' rows.")

df_part3["Waveform_class"] = df_part3["Waveform_class"].astype(str)
df_part3["Mean_stress_fraction"] = df_part3["Mean_stress_fraction"].astype(float)
df_part3["Omega_tau"] = df_part3["Omega_tau"].astype(float)

df_primary = df_part3[
    np.isclose(df_part3["Mean_stress_fraction"], PRIMARY_MEAN_STRESS_FRACTION)
].copy()

if len(df_primary) == 0:
    raise ValueError("Part 3 v4 design space contains no primary mean-load rows.")

PRIMARY_PROTOCOL_SCORE_COL = "Balanced_window_restricted_score"
PRIMARY_PROTOCOL_SCORE_NORM_COL = "Balanced_window_restricted_score_norm"
PRIMARY_PROTOCOL_SCORE_LABEL = "Balanced hard-window multi-objective score"

df_part2_summary.to_csv(os.path.join(OUTDIR, "part4_v4_imported_part2_v3_summary.csv"), index=False)
df_part2_full.to_csv(os.path.join(OUTDIR, "part4_v4_imported_part2_v3_full_window_factor.csv"), index=False)
df_primary.to_csv(os.path.join(OUTDIR, "part4_v4_imported_part3_v4_primary_design_space.csv"), index=False)

input_summary = pd.DataFrame([{
    "Part2_summary_file": PART2_SUMMARY_FILE,
    "Part2_full_window_file": PART2_WINDOW_FILE,
    "Part3_design_space_file": PART3_DESIGN_FILE,
    "Part3_best_file": PART3_BEST_FILE if PART3_BEST_FILE is not None else "Not found",
    "Tau_ref_definition": "eta / E2",
    "Tau_ref_eta_over_E2_seconds": PART2_TAU_REF,
    "Best_omega_tau": BEST_OMEGA_TAU,
    "Window_level": WINDOW_LEVEL,
    "Window_low_omega_tau": WINDOW_LOW,
    "Window_high_omega_tau": WINDOW_HIGH,
    "Primary_mean_stress_fraction": PRIMARY_MEAN_STRESS_FRACTION,
    "Primary_protocol_score": PRIMARY_PROTOCOL_SCORE_LABEL,
    "Interpretation": "Experimental protocol translation; not a clinical prescription."
}])

input_summary.to_csv(os.path.join(OUTDIR, "part4_v4_input_summary.csv"), index=False)

# ============================================================
# 4. Frequency and period scaling tables
# ============================================================

scaling_records = []

for tau_seconds in TAU_RANGE_SECONDS:
    scaling_records.append({
        "Measured_or_assumed_tau_seconds": tau_seconds,
        "Best_omega_tau": BEST_OMEGA_TAU,
        "Window_low_omega_tau": WINDOW_LOW,
        "Window_high_omega_tau": WINDOW_HIGH,
        "Best_frequency_Hz": frequency_from_omega_tau(BEST_OMEGA_TAU, tau_seconds),
        "Window_low_frequency_Hz": frequency_from_omega_tau(WINDOW_LOW, tau_seconds),
        "Window_high_frequency_Hz": frequency_from_omega_tau(WINDOW_HIGH, tau_seconds),
        "Best_period_seconds": period_from_omega_tau(BEST_OMEGA_TAU, tau_seconds),
        "Window_short_period_seconds": period_from_omega_tau(WINDOW_HIGH, tau_seconds),
        "Window_long_period_seconds": period_from_omega_tau(WINDOW_LOW, tau_seconds),
    })

df_scaling = pd.DataFrame(scaling_records)

df_scaling.to_csv(
    os.path.join(OUTDIR, "part4_v4_frequency_period_scaling_tau_1_to_100s.csv"),
    index=False
)

anchor_records = []

for tau_seconds in TAU_ANCHORS_SECONDS:
    anchor_records.append({
        "Measured_or_assumed_tau_seconds": tau_seconds,
        "Best_omega_tau": BEST_OMEGA_TAU,
        "Best_frequency_Hz": frequency_from_omega_tau(BEST_OMEGA_TAU, tau_seconds),
        "Best_period_seconds": period_from_omega_tau(BEST_OMEGA_TAU, tau_seconds),
        "Window_low_frequency_Hz": frequency_from_omega_tau(WINDOW_LOW, tau_seconds),
        "Window_high_frequency_Hz": frequency_from_omega_tau(WINDOW_HIGH, tau_seconds),
        "Window_short_period_seconds": period_from_omega_tau(WINDOW_HIGH, tau_seconds),
        "Window_long_period_seconds": period_from_omega_tau(WINDOW_LOW, tau_seconds),
    })

df_anchor = pd.DataFrame(anchor_records)

df_anchor.to_csv(
    os.path.join(OUTDIR, "part4_v4_tau_anchor_frequency_lookup.csv"),
    index=False
)

# ============================================================
# 5. Translate Part 3 v4 scores into protocol table
# ============================================================

protocol_records = []

for tau_seconds in TAU_ANCHORS_SECONDS:
    for _, row in df_primary.iterrows():
        freq = float(frequency_from_omega_tau(row["Omega_tau"], tau_seconds))
        period = float(period_from_omega_tau(row["Omega_tau"], tau_seconds))

        for session_minutes in SESSION_DURATION_MINUTES:
            cycles_per_session = freq * session_minutes * 60.0

            for daily_sessions in DAILY_SESSIONS:
                daily_cycles = cycles_per_session * daily_sessions
                weekly_cycles = daily_cycles * 7.0

                protocol_records.append({
                    "Design_ID": row["Design_ID"],
                    "Waveform_class": row["Waveform_class"],
                    "Display_name": row["Display_name"],
                    "Mean_stress_fraction": row["Mean_stress_fraction"],
                    "Omega_tau": row["Omega_tau"],
                    "Measured_or_assumed_tau_seconds": tau_seconds,
                    "Frequency_Hz": freq,
                    "Period_seconds": period,
                    "Session_duration_minutes": session_minutes,
                    "Daily_sessions": daily_sessions,
                    "Cycles_per_session": cycles_per_session,
                    "Daily_cycles": daily_cycles,
                    "Weekly_cycles": weekly_cycles,

                    "Part2_v3_window_factor": row["Part2_v3_window_factor"],
                    "In_part2_v3_90pct_window": row["In_part2_v3_90pct_window"],

                    "Global_exploratory_ISI_peak_ratio": row["Global_exploratory_ISI_peak_ratio"],
                    "Global_exploratory_ISI_peak_ratio_norm": row["Global_exploratory_ISI_peak_ratio_norm"],

                    "Window_constrained_ISI_peak_ratio": row["Window_constrained_ISI_peak_ratio"],
                    "Window_constrained_ISI_peak_ratio_norm": row["Window_constrained_ISI_peak_ratio_norm"],

                    "Mechanical_risk_surrogate": row["Mechanical_risk_surrogate"],
                    "Mechanical_safety_surrogate": row["Mechanical_safety_surrogate"],
                    "Smoothness_score": row["Smoothness_score"],

                    "Balanced_window_restricted_score": row["Balanced_window_restricted_score"],
                    "Balanced_window_restricted_score_norm": row["Balanced_window_restricted_score_norm"],

                    "Primary_protocol_score": row[PRIMARY_PROTOCOL_SCORE_COL],
                    "Primary_protocol_score_norm": row[PRIMARY_PROTOCOL_SCORE_NORM_COL],

                    "ISI_raw": row["ISI_raw"],
                    "Peak_abs_strain": row["Peak_abs_strain"],
                    "ISI_peak_strain_ratio": row["ISI_peak_strain_ratio"],
                    "Harmonic_complexity": row["Harmonic_complexity"],
                    "High_frequency_power_fraction": row["High_frequency_power_fraction"],
                    "Very_high_frequency_power_fraction": row["Very_high_frequency_power_fraction"],
                })

df_protocol = pd.DataFrame(protocol_records)

df_protocol["Experimental_protocol_score"] = df_protocol["Primary_protocol_score"]
df_protocol["Experimental_protocol_score_norm"] = minmax_normalize(
    df_protocol["Experimental_protocol_score"]
)

df_protocol.to_csv(
    os.path.join(OUTDIR, "part4_v4_full_experimental_protocol_translation.csv"),
    index=False
)

best_protocol_by_tau = (
    df_protocol
    .sort_values("Experimental_protocol_score", ascending=False)
    .groupby("Measured_or_assumed_tau_seconds")
    .head(1)
    .reset_index(drop=True)
)

best_protocol_by_tau.to_csv(
    os.path.join(OUTDIR, "part4_v4_best_protocol_by_tau.csv"),
    index=False
)

best_protocol_by_tau_session = (
    df_protocol
    .sort_values("Experimental_protocol_score", ascending=False)
    .groupby(["Measured_or_assumed_tau_seconds", "Session_duration_minutes", "Daily_sessions"])
    .head(1)
    .reset_index(drop=True)
)

best_protocol_by_tau_session.to_csv(
    os.path.join(OUTDIR, "part4_v4_best_protocol_by_tau_session.csv"),
    index=False
)

# ============================================================
# 6. Conservative sine reference protocol
# ============================================================

sine_rows = df_protocol[df_protocol["Waveform_class"] == "Sine"].copy()

if len(sine_rows) > 0:
    nearest_sine_omega = sine_rows.iloc[
        np.argmin(np.abs(sine_rows["Omega_tau"].values - BEST_OMEGA_TAU))
    ]["Omega_tau"]

    sine_reference = sine_rows[
        (np.isclose(sine_rows["Omega_tau"], nearest_sine_omega)) &
        (sine_rows["Session_duration_minutes"] == 20) &
        (sine_rows["Daily_sessions"] == 1)
    ].copy()
else:
    sine_reference = pd.DataFrame()

sine_reference.to_csv(
    os.path.join(OUTDIR, "part4_v4_conservative_sine_reference_protocol.csv"),
    index=False
)

# ============================================================
# 7. Experimental validation matrix
# ============================================================

validation_omega_conditions = [
    ("Below window", WINDOW_LOW / 3.0),
    ("Window center", BEST_OMEGA_TAU),
    ("Above window", WINDOW_HIGH * 3.0),
]

validation_records = []

for tau_seconds in VALIDATION_TAU_SECONDS:
    for zone_label, target_omega_tau in validation_omega_conditions:
        for cls in VALIDATION_WAVEFORMS:
            sub = df_primary[df_primary["Waveform_class"] == cls].copy()

            if len(sub) == 0:
                continue

            nearest_idx = int(np.argmin(np.abs(sub["Omega_tau"].values - target_omega_tau)))
            row = sub.iloc[nearest_idx].copy()

            freq = float(frequency_from_omega_tau(row["Omega_tau"], tau_seconds))
            period = float(period_from_omega_tau(row["Omega_tau"], tau_seconds))

            validation_records.append({
                "Measured_or_assumed_tau_seconds": tau_seconds,
                "Frequency_zone": zone_label,
                "Target_omega_tau": target_omega_tau,
                "Actual_nearest_omega_tau": row["Omega_tau"],
                "Waveform_class": row["Waveform_class"],
                "Display_name": row["Display_name"],
                "Frequency_Hz": freq,
                "Period_seconds": period,

                "Part2_v3_window_factor": row["Part2_v3_window_factor"],
                "In_part2_v3_90pct_window": row["In_part2_v3_90pct_window"],

                "Global_exploratory_ISI_peak_ratio_norm": row["Global_exploratory_ISI_peak_ratio_norm"],
                "Window_constrained_ISI_peak_ratio_norm": row["Window_constrained_ISI_peak_ratio_norm"],
                "Balanced_window_restricted_score_norm": row["Balanced_window_restricted_score_norm"],

                "Mechanical_risk_surrogate": row["Mechanical_risk_surrogate"],
                "Mechanical_safety_surrogate": row["Mechanical_safety_surrogate"],
                "Smoothness_score": row["Smoothness_score"],

                "Primary_protocol_score": row[PRIMARY_PROTOCOL_SCORE_COL],
                "Primary_protocol_score_norm": row[PRIMARY_PROTOCOL_SCORE_NORM_COL],
            })

df_validation = pd.DataFrame(validation_records)

df_validation["Validation_score_norm"] = minmax_normalize(
    df_validation["Primary_protocol_score"]
)

df_validation.to_csv(
    os.path.join(OUTDIR, "part4_v4_experimental_validation_matrix.csv"),
    index=False
)

# ============================================================
# 8. Non-sine alternative table
# ============================================================

best_by_omega = (
    df_primary
    .sort_values("Balanced_window_restricted_score", ascending=False)
    .groupby("Omega_tau")
    .head(1)
    .reset_index(drop=True)
)

best_nonsine_by_omega = (
    df_primary[
        (df_primary["Waveform_class"] != "Sine") &
        (df_primary["In_part2_v3_90pct_window"])
    ]
    .sort_values("Balanced_window_restricted_score", ascending=False)
    .groupby("Omega_tau")
    .head(1)
    .reset_index(drop=True)
)

alt_records = []

for _, row_best in best_by_omega.iterrows():
    omega_tau = row_best["Omega_tau"]

    row_alt = best_nonsine_by_omega[
        np.isclose(best_nonsine_by_omega["Omega_tau"], omega_tau)
    ]

    if len(row_alt) == 0:
        continue

    row_alt = row_alt.iloc[0]

    alt_records.append({
        "Omega_tau": omega_tau,
        "Best_balanced_class": row_best["Waveform_class"],
        "Best_balanced_design": row_best["Display_name"],
        "Best_balanced_score": row_best["Balanced_window_restricted_score"],
        "Best_nonsine_balanced_class": row_alt["Waveform_class"],
        "Best_nonsine_balanced_design": row_alt["Display_name"],
        "Best_nonsine_balanced_score": row_alt["Balanced_window_restricted_score"],
        "Best_nonsine_ratio_to_balanced_best": row_alt["Balanced_window_restricted_score"] / max(row_best["Balanced_window_restricted_score"], EPS),
        "In_part2_v3_90pct_window": row_best["In_part2_v3_90pct_window"],
    })

df_alt = pd.DataFrame(alt_records)

df_alt.to_csv(
    os.path.join(OUTDIR, "part4_v4_best_nonsine_alternative_by_omega_tau.csv"),
    index=False
)

# ============================================================
# 9. Cycle-count lookup
# ============================================================

cycle_lookup_records = []

for tau_seconds in TAU_ANCHORS_SECONDS:
    best_freq = float(frequency_from_omega_tau(BEST_OMEGA_TAU, tau_seconds))
    best_period = float(period_from_omega_tau(BEST_OMEGA_TAU, tau_seconds))

    for session_minutes in SESSION_DURATION_MINUTES:
        cycles_per_session = best_freq * session_minutes * 60.0

        cycle_lookup_records.append({
            "Measured_or_assumed_tau_seconds": tau_seconds,
            "Omega_tau": BEST_OMEGA_TAU,
            "Frequency_Hz": best_freq,
            "Period_seconds": best_period,
            "Session_duration_minutes": session_minutes,
            "Cycles_per_session": cycles_per_session,
        })

df_cycle_lookup = pd.DataFrame(cycle_lookup_records)

df_cycle_lookup.to_csv(
    os.path.join(OUTDIR, "part4_v4_cycle_count_lookup.csv"),
    index=False
)

# ============================================================
# 10. Experimental hypothesis table
# ============================================================

top_by_class = (
    df_primary
    .sort_values("Balanced_window_restricted_score", ascending=False)
    .groupby("Waveform_class")
    .head(1)
    .reset_index(drop=True)
)

hypothesis_records = []

for _, row in top_by_class.iterrows():
    hypothesis_records.append({
        "Waveform_class": row["Waveform_class"],
        "Representative_design": row["Display_name"],
        "Omega_tau": row["Omega_tau"],
        "In_part2_v3_90pct_window": row["In_part2_v3_90pct_window"],

        "Global_exploratory_efficiency_norm": row["Global_exploratory_ISI_peak_ratio_norm"],
        "Window_restricted_efficiency_norm": row["Window_constrained_ISI_peak_ratio_norm"],
        "Mechanical_risk_surrogate": row["Mechanical_risk_surrogate"],
        "Mechanical_safety_surrogate": row["Mechanical_safety_surrogate"],
        "Smoothness_score": row["Smoothness_score"],
        "Balanced_score_norm": row["Balanced_window_restricted_score_norm"],

        "Suggested_in_vitro_readouts": "PDL cell alignment, COL1A1, POSTN, RUNX2, ALPL, RANKL/OPG, PTGS2, IL6, TNFA, MMPs, apoptosis",
        "Suggested_in_vivo_readouts": "tooth movement, PDL width, hyalinization, TRAP-positive osteoclast distribution, root resorption lacunae, alveolar bone remodeling",
        "Interpretation_boundary": "Experimental hypothesis only; not a biological proof or clinical prescription."
    })

df_hypothesis = pd.DataFrame(hypothesis_records)

df_hypothesis.to_csv(
    os.path.join(OUTDIR, "part4_v4_experimental_hypothesis_table.csv"),
    index=False
)

# ============================================================
# 11. Decision, interpretation_note, and score-support tables
# ============================================================

decision_table = pd.DataFrame([
    {
        "Decision_domain": "Part 4 v4 role",
        "Result": "Experimental protocol translation",
        "Interpretation": "Converts dimensionless model outputs into testable experimental frequencies, periods, and cycle counts."
    },
    {
        "Decision_domain": "Input dependency",
        "Result": "Reads Part 2 v3 and Part 3 v4 outputs",
        "Interpretation": "Part 4 does not recalculate frequency windows or waveform scores."
    },
    {
        "Decision_domain": "Dimensionless target",
        "Result": f"Best omega_tau = {BEST_OMEGA_TAU:.4g}; 90% window = {WINDOW_LOW:.4g} to {WINDOW_HIGH:.4g}",
        "Interpretation": "Model-derived candidate window, not a measured PDL constant."
    },
    {
        "Decision_domain": "Absolute frequency",
        "Result": "f = omega_tau / (2*pi*tau)",
        "Interpretation": "Experimental Hz depends on measured or assumed tissue/construct tau."
    },
    {
        "Decision_domain": "Primary protocol score",
        "Result": PRIMARY_PROTOCOL_SCORE_LABEL,
        "Interpretation": "Uses Part 3 v4 balanced score rather than single-output mechanical efficiency alone."
    },
    {
        "Decision_domain": "Clinical limitation",
        "Result": "No direct clinical prescription",
        "Interpretation": "Outputs are for PDL-on-a-chip, mechanical testing, and animal-loading protocol design."
    },
])

decision_table.to_csv(os.path.join(OUTDIR, "part4_v4_decision_table.csv"), index=False)

interpretation_note_table = pd.DataFrame([
    {
        "Rule": "Do not claim clinical prescription",
        "Meaning": "Part 4 v4 only translates model outputs into experimental protocol candidates."
    },
    {
        "Rule": "Do not claim universal Hz",
        "Meaning": "Absolute frequency depends on measured tau."
    },
    {
        "Rule": "Use balanced score as primary protocol score",
        "Meaning": "Protocol ranking should not be based on stimulation efficiency alone."
    },
    {
        "Rule": "Do not recalculate Part 2 or Part 3",
        "Meaning": "Part 4 v4 must read upstream outputs."
    },
    {
        "Rule": "Clinical force-decay is not modeled here",
        "Meaning": "Current Part 4 translates idealized dynamic waveform archetypes; clinical-like stepped relaxation should be treated separately."
    },
])

interpretation_note_table.to_csv(os.path.join(OUTDIR, "part4_v4_interpretation_interpretation_notes.csv"), index=False)


# ============================================================
# 12. Figures A-J
# ============================================================

# A. Absolute frequency scaling
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
ax.set_xlabel("Measured or assumed τ (s)")
ax.set_ylabel("Absolute frequency (Hz)")
ax.set_title("Dimensionless window translated to experimental frequency", fontsize=9, loc="left")
ax.legend(fontsize=5.8, loc="upper right")
panel_label(ax, "A")
save_fig(fig, "part4_v4_panel_A_absolute_frequency_scaling.png")

# B. Period scaling
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

ax.plot(
    df_scaling["Measured_or_assumed_tau_seconds"],
    df_scaling["Best_period_seconds"],
    color="#4A6572",
    linewidth=2.0,
    label="Best period"
)

ax.fill_between(
    df_scaling["Measured_or_assumed_tau_seconds"],
    df_scaling["Window_short_period_seconds"],
    df_scaling["Window_long_period_seconds"],
    color="#7B9EA6",
    alpha=0.18,
    label="90% window"
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Measured or assumed τ (s)")
ax.set_ylabel("Loading period (s)")
ax.set_title("Experimental loading period scales with τ", fontsize=9, loc="left")
ax.legend(fontsize=5.8, loc="upper left")
panel_label(ax, "B")
save_fig(fig, "part4_v4_panel_B_period_scaling.png")

# C. Balanced protocol map
protocol_map = df_protocol[
    (df_protocol["Session_duration_minutes"] == 20) &
    (df_protocol["Daily_sessions"] == 1)
].copy()

best_map = (
    protocol_map
    .sort_values("Experimental_protocol_score", ascending=False)
    .groupby(["Measured_or_assumed_tau_seconds", "Omega_tau"])
    .head(1)
    .reset_index(drop=True)
)

pivot_score = (
    best_map
    .pivot_table(
        index="Measured_or_assumed_tau_seconds",
        columns="Omega_tau",
        values="Experimental_protocol_score_norm",
        aggfunc="max"
    )
    .sort_index()
)

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

im = ax.imshow(
    pivot_score.values,
    origin="lower",
    aspect="auto",
    cmap=CMAP_SEQ,
    vmin=0,
    vmax=1,
    extent=[
        np.log10(pivot_score.columns.min()),
        np.log10(pivot_score.columns.max()),
        np.log10(pivot_score.index.min()),
        np.log10(pivot_score.index.max())
    ]
)

ax.axvspan(np.log10(WINDOW_LOW), np.log10(WINDOW_HIGH), color="#FFFFFF", alpha=0.20)
ax.set_xlabel(r"$\log_{10}(\omega\tau)$")
ax.set_ylabel(r"$\log_{10}(\tau\ \mathrm{s})$")
ax.set_title("Balanced experimental protocol map", fontsize=9, loc="left")

cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
cbar.set_label("Normalized balanced score", fontsize=6.5)
cbar.ax.tick_params(labelsize=5.8)
panel_label(ax, "C")
save_fig(fig, "part4_v4_panel_C_balanced_protocol_map.png")

# D. Best class by tau
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

for cls in CLASS_ORDER:
    sub = best_protocol_by_tau[best_protocol_by_tau["Waveform_class"] == cls]
    if len(sub) == 0:
        continue

    ax.scatter(
        sub["Measured_or_assumed_tau_seconds"],
        sub["Frequency_Hz"],
        color=CLASS_COLORS[cls],
        s=58,
        edgecolor="white",
        linewidth=0.7,
        label=cls
    )

ax.plot(
    df_scaling["Measured_or_assumed_tau_seconds"],
    df_scaling["Best_frequency_Hz"],
    color="#2d2d2d",
    linestyle=":",
    linewidth=1.2,
    label=r"$\omega\tau$ scaling"
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Measured or assumed τ (s)")
ax.set_ylabel("Experimental frequency (Hz)")
ax.set_title("Best balanced experimental class across τ", fontsize=9, loc="left")
ax.legend(fontsize=5.5, loc="upper right", ncol=2)
panel_label(ax, "D")
save_fig(fig, "part4_v4_panel_D_best_balanced_class_by_tau.png")

# E. Top protocol candidates at tau = 20 s
tau_for_top = 20.0

df_top = (
    df_protocol[
        (np.isclose(df_protocol["Measured_or_assumed_tau_seconds"], tau_for_top)) &
        (df_protocol["Session_duration_minutes"] == 20) &
        (df_protocol["Daily_sessions"] == 1)
    ]
    .sort_values("Experimental_protocol_score", ascending=False)
    .head(10)
    .copy()
)

df_top["Short_label"] = (
    df_top["Display_name"] +
    "\n" +
    df_top["Omega_tau"].map(lambda x: f"ωτ={x:.1f}")
)

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

bars = ax.bar(
    np.arange(len(df_top)),
    df_top["Experimental_protocol_score_norm"],
    color=[CLASS_COLORS[x] for x in df_top["Waveform_class"]],
    edgecolor="white",
    linewidth=0.6,
    width=0.68
)

ax.set_xticks(np.arange(len(df_top)))
ax.set_xticklabels(df_top["Short_label"], rotation=35, ha="right", fontsize=5.2)
ax.set_ylabel("Normalized balanced score")
ax.set_title("Top balanced protocol candidates at τ = 20 s", fontsize=9, loc="left")
ax.set_ylim(0, 1.12)

for bar, val in zip(bars, df_top["Experimental_protocol_score_norm"]):
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
save_fig(fig, "part4_v4_panel_E_top_balanced_protocol_candidates_tau20.png")

# F. Cycle-count heatmap
cycle_pivot = (
    df_cycle_lookup
    .pivot_table(
        index="Measured_or_assumed_tau_seconds",
        columns="Session_duration_minutes",
        values="Cycles_per_session",
        aggfunc="mean"
    )
    .sort_index()
)

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

im = ax.imshow(
    cycle_pivot.values,
    origin="lower",
    aspect="auto",
    cmap=CMAP_SEQ,
    extent=[
        cycle_pivot.columns.min(),
        cycle_pivot.columns.max(),
        np.log10(cycle_pivot.index.min()),
        np.log10(cycle_pivot.index.max())
    ]
)

ax.set_xlabel("Session duration (min)")
ax.set_ylabel(r"$\log_{10}(\tau\ \mathrm{s})$")
ax.set_title("Cycle count at the Part 2 v3 window center", fontsize=9, loc="left")

cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
cbar.set_label("Cycles per session", fontsize=6.5)
cbar.ax.tick_params(labelsize=5.8)
panel_label(ax, "F")
save_fig(fig, "part4_v4_panel_F_cycle_count_lookup_heatmap.png")

# G. Balanced validation matrix
validation_pivot = (
    df_validation
    .pivot_table(
        index="Waveform_class",
        columns="Frequency_zone",
        values="Validation_score_norm",
        aggfunc="mean"
    )
    .reindex(
        index=["Sine", "Triangle", "Asymmetric", "Square", "Pulse"],
        columns=["Below window", "Window center", "Above window"]
    )
)

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

im = ax.imshow(
    validation_pivot.values,
    cmap=CMAP_SEQ,
    aspect="auto",
    vmin=0,
    vmax=1
)

ax.set_xticks(np.arange(validation_pivot.shape[1]))
ax.set_xticklabels(validation_pivot.columns, fontsize=5.8, rotation=18, ha="right")
ax.set_yticks(np.arange(validation_pivot.shape[0]))
ax.set_yticklabels(validation_pivot.index, fontsize=6.0)
ax.set_title("Balanced experimental validation matrix", fontsize=9, loc="left")

for i in range(validation_pivot.shape[0]):
    for j in range(validation_pivot.shape[1]):
        v = validation_pivot.values[i, j]
        if np.isfinite(v):
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=5.6,
                color="white" if v > 0.60 else "#2d2d2d"
            )

cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
cbar.set_label("Normalized balanced score", fontsize=6.5)
cbar.ax.tick_params(labelsize=5.8)
panel_label(ax, "G")
save_fig(fig, "part4_v4_panel_G_balanced_validation_matrix.png")

# H. Non-sine balanced alternative ratio
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

if len(df_alt) > 0:
    ax.plot(
        df_alt["Omega_tau"],
        df_alt["Best_nonsine_ratio_to_balanced_best"],
        color="#4A6572",
        linewidth=1.8,
        label="Best non-sine / balanced best"
    )

ax.axvspan(WINDOW_LOW, WINDOW_HIGH, color="#A67B7B", alpha=0.10, label="Part 2 v3 90% window")
ax.axhline(0.95, color="#2d2d2d", linestyle=":", linewidth=1.0, label="95% near-optimal")

ax.set_xscale("log")
ax.set_ylim(0, 1.08)
ax.set_xlabel(r"Dimensionless frequency, $\omega\tau$")
ax.set_ylabel("Non-sine ratio to balanced best")
ax.set_title("Near-optimal non-sine balanced alternatives", fontsize=9, loc="left")
ax.legend(fontsize=5.8, loc="lower right")
panel_label(ax, "H")
save_fig(fig, "part4_v4_panel_H_nonsine_balanced_alternative_ratio.png")

# I. Risk-smoothness protocol scatter at tau = 20 s
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

plot_protocol = df_protocol[
    (df_protocol["Session_duration_minutes"] == 20) &
    (df_protocol["Daily_sessions"] == 1) &
    (np.isclose(df_protocol["Measured_or_assumed_tau_seconds"], 20.0)) &
    (df_protocol["In_part2_v3_90pct_window"])
].copy()

for cls in CLASS_ORDER:
    sub = plot_protocol[plot_protocol["Waveform_class"] == cls]
    if len(sub) == 0:
        continue

    ax.scatter(
        sub["Mechanical_risk_surrogate"],
        sub["Smoothness_score"],
        s=20 + 70 * sub["Experimental_protocol_score_norm"],
        color=CLASS_COLORS[cls],
        alpha=0.55,
        edgecolor="white",
        linewidth=0.4,
        label=cls
    )

ax.set_xlabel("Mechanical risk surrogate")
ax.set_ylabel("Smoothness score")
ax.set_title("Balanced protocol risk-smoothness map at τ = 20 s", fontsize=9, loc="left")
ax.legend(fontsize=5.2, loc="lower left", ncol=2)
panel_label(ax, "I")
save_fig(fig, "part4_v4_panel_I_risk_smoothness_protocol_scatter.png")

# J. Frequency lookup table
lookup_display = df_anchor[[
    "Measured_or_assumed_tau_seconds",
    "Best_frequency_Hz",
    "Best_period_seconds",
    "Window_low_frequency_Hz",
    "Window_high_frequency_Hz"
]].copy()

table_data = lookup_display.copy()

for col in table_data.columns:
    table_data[col] = table_data[col].map(lambda x: f"{x:.3g}")

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)
ax.axis("off")

table = ax.table(
    cellText=table_data.values,
    colLabels=["τ (s)", "Best Hz", "Best period", "Low Hz", "High Hz"],
    loc="center",
    cellLoc="center",
    colLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(5.8)
table.scale(1.0, 1.35)

for _, cell in table.get_celld().items():
    cell.set_edgecolor("#D0D0D0")
    cell.set_linewidth(0.4)

ax.set_title("Experimental frequency lookup table", fontsize=9, loc="left")
panel_label(ax, "J")
save_fig(fig, "part4_v4_panel_J_frequency_lookup_table.png")

# ============================================================
# 13. Combined 600 dpi overview
# ============================================================

def create_combined_overview_600dpi():
    if not PIL_AVAILABLE:
        print("Pillow is not available. Combined overview was skipped. Install pillow if needed: pip install pillow")
        return

    panel_files = [
        "part4_v4_panel_A_absolute_frequency_scaling.png",
        "part4_v4_panel_B_period_scaling.png",
        "part4_v4_panel_C_balanced_protocol_map.png",
        "part4_v4_panel_D_best_balanced_class_by_tau.png",
        "part4_v4_panel_E_top_balanced_protocol_candidates_tau20.png",
        "part4_v4_panel_F_cycle_count_lookup_heatmap.png",
        "part4_v4_panel_G_balanced_validation_matrix.png",
        "part4_v4_panel_H_nonsine_balanced_alternative_ratio.png",
        "part4_v4_panel_I_risk_smoothness_protocol_scatter.png",
        "part4_v4_panel_J_frequency_lookup_table.png",
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

    out_path = os.path.join(OUTDIR, "part4_v4_combined_overview_600dpi.png")
    canvas.save(out_path, dpi=(600, 600))

    out_path_alias = os.path.join(OUTDIR, "part4_v4_combined_overview.png")
    canvas.save(out_path_alias, dpi=(600, 600))

    print(f"Combined 600 dpi overview saved to: {out_path}")
    print(f"Combined 600 dpi overview alias saved to: {out_path_alias}")
    print(f"Combined overview includes {len(panel_paths)} panels.")

create_combined_overview_600dpi()

# ============================================================
# 14. Summary report
# ============================================================

print("\n" + "=" * 90)
print("Part 4 protocol-translation analysis completed.")
print("=" * 90)
print("Script filename recommendation:")
print("part4_protocol_translation.py")
print("")
print("Output folder:")
print(OUTDIR)
print("")
print("Required inputs:")
print(PART2_SUMMARY_FILE)
print(PART2_WINDOW_FILE)
print(PART3_DESIGN_FILE)
print("")
print("Hard chain rule:")
print("Part 4 v4 reads manuscript-linked Part 2 v3 and Part 3 v4 outputs. It does not recalculate upstream windows or waveform scores.")
print("")
print("Primary protocol score:")
print(PRIMARY_PROTOCOL_SCORE_LABEL)
print("")
print("Input summary:")
print(input_summary.to_string(index=False))
print("")
print("Decision table:")
print(decision_table.to_string(index=False))
print("")
print("Frequency anchor lookup:")
print(df_anchor.to_string(index=False))
print("")
print("Best protocol by tau:")
print(best_protocol_by_tau[[
    "Measured_or_assumed_tau_seconds",
    "Waveform_class",
    "Display_name",
    "Omega_tau",
    "Frequency_Hz",
    "Period_seconds",
    "Experimental_protocol_score_norm",
    "Mechanical_risk_surrogate",
    "Smoothness_score"
]].to_string(index=False))
print("")
print("Validation matrix preview:")
print(df_validation[[
    "Measured_or_assumed_tau_seconds",
    "Frequency_zone",
    "Waveform_class",
    "Actual_nearest_omega_tau",
    "Frequency_Hz",
    "Period_seconds",
    "Validation_score_norm",
    "Mechanical_risk_surrogate",
    "Smoothness_score"
]].head(25).to_string(index=False))
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
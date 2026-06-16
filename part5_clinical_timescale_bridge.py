# ============================================================
# Part 5 MANUSCRIPT v3 Manuscript version:
# Clinical Timescale Bridge
#
# Required previous steps:
# 1. Run part2_frequency_window.py
# 2. Run part3_waveform_screening.py
#
# Scientific role:
# This part addresses a translational criticism:
# clinical-scale loading can involve force decay and periodic renewal
# over days to weeks.
#
# Central conclusion:
# Clinical force renewal occurs at appointment-scale time intervals,
# whereas the Part 2-4 framework addresses dynamic PDL-scale stimulation
# over seconds to minutes.
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
# 0. Output folder and figure style
# ============================================================

OUTDIR = "part5_clinical_timescale_bridge_outputs"
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
    "Ideal static reference": "#D9E3E5",
    "28-day renewal": "#8A9E82",
    "14-day renewal": "#B5A07A",
    "7-day renewal": "#7B9EA6",
    "4-day renewal": "#8E85A6",
    "1-day renewal": "#A67B7B",
    "Dynamic-window center": "#4A6572",
    "Balanced dynamic candidate": "#2d2d2d",
}

CMAP_SEQ = mcolors.LinearSegmentedColormap.from_list(
    "morandi_seq",
    ["#EEF3F4", "#C8D8DF", "#7B9EA6", "#4A6572", "#A67B7B"]
)

CMAP_DIV = mcolors.LinearSegmentedColormap.from_list(
    "morandi_div",
    ["#EEF3F4", "#D9E3E5", "#7B9EA6", "#4A6572"]
)

EPS = 1e-12

# ============================================================
# 1. Helper functions
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

def require_columns(df, required_cols, source_name):
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"{source_name} is missing required columns: {', '.join(missing)}")

def frequency_from_omega_tau(omega_tau, tau_seconds):
    return np.asarray(omega_tau, dtype=float) / (2.0 * np.pi * np.asarray(tau_seconds, dtype=float))

def period_from_omega_tau(omega_tau, tau_seconds):
    freq = frequency_from_omega_tau(omega_tau, tau_seconds)
    return 1.0 / np.maximum(freq, EPS)

def omega_tau_from_period_seconds(period_seconds, tau_seconds):
    return 2.0 * np.pi * np.asarray(tau_seconds, dtype=float) / np.asarray(period_seconds, dtype=float)

def day_to_seconds(days):
    return days * 24.0 * 3600.0

def seconds_to_days(seconds):
    return seconds / (24.0 * 3600.0)

def periodic_exp_decay_force(t_days, interval_days, decay_tau_days, f0=1.0, floor=0.05):
    t_days = np.asarray(t_days, dtype=float)
    age = np.mod(t_days, interval_days)
    return floor + (f0 - floor) * np.exp(-age / max(decay_tau_days, EPS))

def normalize_to_initial(force):
    force = np.asarray(force, dtype=float)
    return force / max(force[0], EPS)

def safe_log10_ratio(numerator, denominator):
    numerator = float(numerator)
    denominator = float(denominator)
    if numerator <= 0 or denominator <= 0:
        return np.nan
    return float(np.log10(numerator / denominator))

# ============================================================
# 2. Locate and read Part 2 v3 / Part 3 v4
# ============================================================

def locate_part2_v3_summary_file():
    candidate = os.path.join(
        "part2_frequency_window_outputs",
        "part2_v3_baseline_full_model_window_summary.csv"
    )
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        "Cannot find Part 2 v3 summary file. Run Part 2 v3 first."
    )

def locate_part2_v3_full_window_file():
    candidate = os.path.join(
        "part2_frequency_window_outputs",
        "part2_v3_frequency_response_all_ablation_modes.csv"
    )
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        "Cannot find Part 2 v3 frequency-response file. Run Part 2 v3 first."
    )

def locate_part3_v4_balanced_best_file():
    candidate = os.path.join(
        "part3_waveform_screening_outputs",
        "part3_v4_balanced_best_primary_design.csv"
    )
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        "Cannot find Part 3 v4 balanced-best file. Run Part 3 v4 first."
    )

PART2_SUMMARY_FILE = locate_part2_v3_summary_file()
PART2_WINDOW_FILE = locate_part2_v3_full_window_file()
PART3_BALANCED_FILE = locate_part3_v4_balanced_best_file()

df_part2_summary = pd.read_csv(PART2_SUMMARY_FILE)
df_part2_all = pd.read_csv(PART2_WINDOW_FILE)
df_part3_balanced = pd.read_csv(PART3_BALANCED_FILE)

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

require_columns(
    df_part3_balanced,
    [
        "Display_name",
        "Waveform_class",
        "Omega_tau",
        "Balanced_window_restricted_score_norm",
        "Mechanical_risk_surrogate",
        "Smoothness_score"
    ],
    "Part 3 v4 balanced-best file"
)

if str(df_part2_summary["Tau_ref_definition"].iloc[0]).strip() != "eta / E2":
    raise ValueError("Part 2 v3 summary does not report tau_ref = eta / E2.")

PART2_BEST_OMEGA_TAU = float(df_part2_summary["Best_omega_tau"].iloc[0])
PART2_WINDOW_LOW = float(df_part2_summary["Window_low_omega_tau"].iloc[0])
PART2_WINDOW_HIGH = float(df_part2_summary["Window_high_omega_tau"].iloc[0])
BASE_TAU_REF_SECONDS = float(df_part2_summary["Tau_ref_eta_over_E2_seconds"].iloc[0])

PART3_BALANCED_OMEGA_TAU = float(df_part3_balanced["Omega_tau"].iloc[0])
PART3_BALANCED_WAVEFORM = str(df_part3_balanced["Display_name"].iloc[0])
PART3_BALANCED_CLASS = str(df_part3_balanced["Waveform_class"].iloc[0])

df_part2_full = df_part2_all[df_part2_all["Mode"] == "full"].copy()
df_part2_full = df_part2_full.sort_values("Omega_tau").reset_index(drop=True)

if len(df_part2_full) == 0:
    raise ValueError("Part 2 v3 window file has no Mode == 'full' rows.")

def interpolate_part2_window_factor(omega_tau_values):
    omega_tau_values = np.asarray(omega_tau_values, dtype=float)
    x = np.log10(df_part2_full["Omega_tau"].values)
    y = df_part2_full["Normalized_score"].values

    return np.interp(
        np.log10(np.maximum(omega_tau_values, 1e-300)),
        x,
        y,
        left=y[0],
        right=y[-1]
    )

df_part2_summary.to_csv(os.path.join(OUTDIR, "part5_v3_imported_part2_v3_summary.csv"), index=False)
df_part2_full.to_csv(os.path.join(OUTDIR, "part5_v3_imported_part2_v3_full_window.csv"), index=False)
df_part3_balanced.to_csv(os.path.join(OUTDIR, "part5_v3_imported_part3_v4_balanced_best.csv"), index=False)

# ============================================================
# 3. Representative clinical force-renewal profiles
# ============================================================

CLINICAL_PROFILES = [
    {
        "Profile": "Ideal static reference",
        "Category": "Reference",
        "Renewal_interval_days": np.inf,
        "Decay_tau_days": np.inf,
        "Initial_force": 1.0,
        "Floor_force": 1.0,
        "Description": "Idealized non-decaying reference used for model comparison."
    },
    {
        "Profile": "28-day renewal",
        "Category": "Clinical-scale renewal",
        "Renewal_interval_days": 28.0,
        "Decay_tau_days": 10.0,
        "Initial_force": 1.0,
        "Floor_force": 0.10,
        "Description": "Representative monthly renewal with gradual force decay."
    },
    {
        "Profile": "14-day renewal",
        "Category": "Clinical-scale renewal",
        "Renewal_interval_days": 14.0,
        "Decay_tau_days": 4.0,
        "Initial_force": 1.0,
        "Floor_force": 0.08,
        "Description": "Representative two-week renewal profile."
    },
    {
        "Profile": "7-day renewal",
        "Category": "Clinical-scale renewal",
        "Renewal_interval_days": 7.0,
        "Decay_tau_days": 2.0,
        "Initial_force": 1.0,
        "Floor_force": 0.06,
        "Description": "Representative weekly renewal profile."
    },
    {
        "Profile": "4-day renewal",
        "Category": "Clinical-scale renewal",
        "Renewal_interval_days": 4.0,
        "Decay_tau_days": 1.2,
        "Initial_force": 1.0,
        "Floor_force": 0.05,
        "Description": "Representative accelerated renewal profile."
    },
    {
        "Profile": "1-day renewal",
        "Category": "Clinical-scale renewal",
        "Renewal_interval_days": 1.0,
        "Decay_tau_days": 0.35,
        "Initial_force": 1.0,
        "Floor_force": 0.05,
        "Description": "Representative daily renewal profile."
    },
]

df_profiles = pd.DataFrame(CLINICAL_PROFILES)
df_profiles.to_csv(os.path.join(OUTDIR, "part5_v3_clinical_profile_definitions.csv"), index=False)

# ============================================================
# 4. Simulate clinical-scale force decay over 56 days
# ============================================================

SIM_DAYS = 56.0
DT_DAYS = 0.02
t_days = np.arange(0.0, SIM_DAYS + DT_DAYS, DT_DAYS)

force_records = []

for profile in CLINICAL_PROFILES:
    name = profile["Profile"]

    if np.isfinite(profile["Renewal_interval_days"]):
        force = periodic_exp_decay_force(
            t_days,
            interval_days=profile["Renewal_interval_days"],
            decay_tau_days=profile["Decay_tau_days"],
            f0=profile["Initial_force"],
            floor=profile["Floor_force"]
        )
    else:
        force = np.ones_like(t_days) * profile["Initial_force"]

    force = normalize_to_initial(force)

    for ti, fi in zip(t_days, force):
        force_records.append({
            "Time_days": ti,
            "Profile": name,
            "Normalized_force": fi,
            "Category": profile["Category"],
        })

df_force = pd.DataFrame(force_records)
df_force.to_csv(os.path.join(OUTDIR, "part5_v3_simulated_clinical_force_decay_profiles.csv"), index=False)

# ============================================================
# 5. Time-scale and omega*tau comparison at baseline tau
# ============================================================

timescale_records = []

for profile in CLINICAL_PROFILES:
    name = profile["Profile"]
    interval_days = profile["Renewal_interval_days"]

    if np.isfinite(interval_days):
        interval_seconds = day_to_seconds(interval_days)
        effective_frequency_hz = 1.0 / interval_seconds
        effective_omega_tau = float(omega_tau_from_period_seconds(interval_seconds, BASE_TAU_REF_SECONDS))
        part2_factor = float(interpolate_part2_window_factor([effective_omega_tau])[0])
        in_window = bool((effective_omega_tau >= PART2_WINDOW_LOW) and (effective_omega_tau <= PART2_WINDOW_HIGH))
        orders_below_window_low = safe_log10_ratio(PART2_WINDOW_LOW, effective_omega_tau)
    else:
        interval_seconds = np.inf
        effective_frequency_hz = 0.0
        effective_omega_tau = 0.0
        part2_factor = 0.0
        in_window = False
        orders_below_window_low = np.inf

    sub = df_force[df_force["Profile"] == name].copy()
    mean_force = float(sub["Normalized_force"].mean())
    min_force = float(sub["Normalized_force"].min())
    max_force = float(sub["Normalized_force"].max())
    force_auc_fraction = float(np.trapz(sub["Normalized_force"], sub["Time_days"]) / SIM_DAYS)

    if np.isfinite(interval_days):
        renewal_count_56d = int(np.floor(SIM_DAYS / interval_days))
        force_at_end_of_cycle = float(
            periodic_exp_decay_force(
                np.array([interval_days - DT_DAYS]),
                interval_days=interval_days,
                decay_tau_days=profile["Decay_tau_days"],
                f0=profile["Initial_force"],
                floor=profile["Floor_force"]
            )[0]
        )
        force_drop_per_cycle = 1.0 - force_at_end_of_cycle
    else:
        renewal_count_56d = 0
        force_drop_per_cycle = 0.0

    if name == "Ideal static reference":
        interpretation = "Idealized reference, not a clinical force-delivery profile."
    else:
        interpretation = "Clinical-scale force renewal; not equivalent to dynamic PDL-scale stimulation."

    timescale_records.append({
        "Profile": name,
        "Category": profile["Category"],
        "Renewal_interval_days": interval_days,
        "Renewal_interval_seconds": interval_seconds,
        "Effective_frequency_Hz": effective_frequency_hz,
        "Effective_omega_tau_at_baseline_tau": effective_omega_tau,
        "Orders_below_part2_window_low": orders_below_window_low,
        "Part2_full_model_factor_at_effective_omega_tau": part2_factor,
        "Inside_part2_window": in_window,
        "Mean_normalized_force_over_56d": mean_force,
        "Min_normalized_force_over_56d": min_force,
        "Max_normalized_force_over_56d": max_force,
        "Force_AUC_fraction_over_56d": force_auc_fraction,
        "Renewal_count_over_56d": renewal_count_56d,
        "Approx_force_drop_per_cycle": force_drop_per_cycle,
        "Interpretation": interpretation
    })

for label, omega_tau in [
    ("Dynamic-window center", PART2_BEST_OMEGA_TAU),
    ("Balanced dynamic candidate", PART3_BALANCED_OMEGA_TAU),
]:
    period_seconds = float(period_from_omega_tau(omega_tau, BASE_TAU_REF_SECONDS))
    frequency_hz = float(frequency_from_omega_tau(omega_tau, BASE_TAU_REF_SECONDS))
    part2_factor = float(interpolate_part2_window_factor([omega_tau])[0])
    in_window = bool((omega_tau >= PART2_WINDOW_LOW) and (omega_tau <= PART2_WINDOW_HIGH))

    timescale_records.append({
        "Profile": label,
        "Category": "Dynamic stimulation",
        "Renewal_interval_days": seconds_to_days(period_seconds),
        "Renewal_interval_seconds": period_seconds,
        "Effective_frequency_Hz": frequency_hz,
        "Effective_omega_tau_at_baseline_tau": omega_tau,
        "Orders_below_part2_window_low": np.nan,
        "Part2_full_model_factor_at_effective_omega_tau": part2_factor,
        "Inside_part2_window": in_window,
        "Mean_normalized_force_over_56d": np.nan,
        "Min_normalized_force_over_56d": np.nan,
        "Max_normalized_force_over_56d": np.nan,
        "Force_AUC_fraction_over_56d": np.nan,
        "Renewal_count_over_56d": np.nan,
        "Approx_force_drop_per_cycle": np.nan,
        "Interpretation": "Dynamic loading candidate inside the Part 2 mechanobiological window."
    })

df_timescale = pd.DataFrame(timescale_records)
df_timescale.to_csv(os.path.join(OUTDIR, "part5_v3_macro_vs_micro_timescale_table.csv"), index=False)

# ============================================================
# 6. Tau-sensitivity table for clinical-scale renewal intervals
# ============================================================

TAU_ANCHORS_SECONDS = np.array([1, 2, 3, 5, 10, 20, 30, 60, 100], dtype=float)

macro_tau_records = []

for profile in CLINICAL_PROFILES:
    if not np.isfinite(profile["Renewal_interval_days"]):
        continue

    interval_seconds = day_to_seconds(profile["Renewal_interval_days"])

    for tau_seconds in TAU_ANCHORS_SECONDS:
        effective_omega_tau = float(omega_tau_from_period_seconds(interval_seconds, tau_seconds))
        in_window = bool((effective_omega_tau >= PART2_WINDOW_LOW) and (effective_omega_tau <= PART2_WINDOW_HIGH))
        part2_factor = float(interpolate_part2_window_factor([effective_omega_tau])[0])
        orders_below = safe_log10_ratio(PART2_WINDOW_LOW, effective_omega_tau)

        macro_tau_records.append({
            "Profile": profile["Profile"],
            "Renewal_interval_days": profile["Renewal_interval_days"],
            "Tau_seconds": tau_seconds,
            "Effective_omega_tau": effective_omega_tau,
            "Inside_part2_window": in_window,
            "Orders_below_part2_window_low": orders_below,
            "Part2_full_model_factor": part2_factor,
        })

df_macro_tau = pd.DataFrame(macro_tau_records)
df_macro_tau.to_csv(os.path.join(OUTDIR, "part5_v3_clinical_renewal_tau_sensitivity.csv"), index=False)

# ============================================================
# 7. Required dynamic periods for Part 2 window
# ============================================================

window_period_records = []

for tau_seconds in TAU_ANCHORS_SECONDS:
    window_period_records.append({
        "Tau_seconds": tau_seconds,
        "Part2_best_omega_tau": PART2_BEST_OMEGA_TAU,
        "Part2_best_frequency_Hz": float(frequency_from_omega_tau(PART2_BEST_OMEGA_TAU, tau_seconds)),
        "Part2_best_period_seconds": float(period_from_omega_tau(PART2_BEST_OMEGA_TAU, tau_seconds)),
        "Part2_window_low_omega_tau": PART2_WINDOW_LOW,
        "Part2_window_high_omega_tau": PART2_WINDOW_HIGH,
        "Window_short_period_seconds": float(period_from_omega_tau(PART2_WINDOW_HIGH, tau_seconds)),
        "Window_long_period_seconds": float(period_from_omega_tau(PART2_WINDOW_LOW, tau_seconds)),
        "Part3_balanced_omega_tau": PART3_BALANCED_OMEGA_TAU,
        "Part3_balanced_frequency_Hz": float(frequency_from_omega_tau(PART3_BALANCED_OMEGA_TAU, tau_seconds)),
        "Part3_balanced_period_seconds": float(period_from_omega_tau(PART3_BALANCED_OMEGA_TAU, tau_seconds)),
    })

df_window_period = pd.DataFrame(window_period_records)
df_window_period.to_csv(os.path.join(OUTDIR, "part5_v3_required_microdynamic_periods_by_tau.csv"), index=False)

# ============================================================
# 8. Neutral comparison tables
# ============================================================

scale_comparison_table = pd.DataFrame([
    {
        "Layer": "Ideal static reference",
        "Typical_time_scale": "No renewal",
        "Main_quantity": "Reference force level",
        "Role_in_manuscript": "Model comparator",
        "Relation_to_part2_window": "Not a clinical renewal profile."
    },
    {
        "Layer": "Clinical-scale force renewal",
        "Typical_time_scale": "Days to weeks",
        "Main_quantity": "Force decay and renewal interval",
        "Role_in_manuscript": "Clinical reality bridge",
        "Relation_to_part2_window": "Far below the dynamic window under second-scale tau assumptions."
    },
    {
        "Layer": "Dynamic PDL stimulation",
        "Typical_time_scale": "Seconds to minutes",
        "Main_quantity": "Frequency, waveform, and strain-rate stimulation",
        "Role_in_manuscript": "Primary theoretical framework",
        "Relation_to_part2_window": "Directly defined by omega*tau."
    },
    {
        "Layer": "Experimental validation",
        "Typical_time_scale": "Protocol-dependent",
        "Main_quantity": "Measured cellular and tissue-level readouts",
        "Role_in_manuscript": "Future testable hypotheses",
        "Relation_to_part2_window": "Should compare below-window, window-center, and above-window conditions."
    },
])

scale_comparison_table.to_csv(os.path.join(OUTDIR, "part5_v3_neutral_scale_comparison_table.csv"), index=False)

decision_table = pd.DataFrame([
    {
        "Question": "Does Part 1 constant loading represent real clinical orthodontic force delivery?",
        "Answer": "No.",
        "Manuscript_use": "Constant loading should be described as an idealized static reference."
    },
    {
        "Question": "Do common clinical appliances deliver mathematically constant force?",
        "Answer": "No.",
        "Manuscript_use": "They are better represented as force decay and renewal profiles."
    },
    {
        "Question": "Are clinical renewal intervals equivalent to Part 2 dynamic loading?",
        "Answer": "No.",
        "Manuscript_use": "Clinical renewal occurs over days to weeks and lies far below the dynamic omega*tau window for typical second-scale tau assumptions."
    },
    {
        "Question": "Does this undermine the Part 1-4 framework?",
        "Answer": "No.",
        "Manuscript_use": "It clarifies that the framework addresses dynamic mechanobiological stimulation, not appointment-scale force renewal."
    },
    {
        "Question": "How should this limitation be stated?",
        "Answer": "As a time-scale distinction.",
        "Manuscript_use": "Clinical-scale renewal and dynamic stimulation should be discussed as distinct temporal regimes."
    },
])

decision_table.to_csv(os.path.join(OUTDIR, "part5_v3_decision_table.csv"), index=False)

interpretation_note_table = pd.DataFrame([
    {
        "Rule": "Do not claim that clinical orthodontics is constant force.",
        "Reason": "Common clinical appliances show force decay and renewal."
    },
    {
        "Rule": "Do not equate weekly or monthly renewal with dynamic PDL-scale stimulation.",
        "Reason": "They differ by several orders of magnitude in effective omega*tau."
    },
    {
        "Rule": "Do not convert Part 5 into a clinical prescription.",
        "Reason": "Part 5 is a time-scale bridge and reviewer-defense layer."
    },
    {
        "Rule": "Do not infer biological outcomes from Part 5 alone.",
        "Reason": "Part 5 only compares time scales and model-derived window factors."
    },
    {
        "Rule": "Keep implementation language out of this part.",
        "Reason": "The manuscript only needs the time-scale distinction."
    },
])

interpretation_note_table.to_csv(os.path.join(OUTDIR, "part5_v3_interpretation_interpretation_notes.csv"), index=False)

figure_recommendation_table = pd.DataFrame([
    {
        "Panel": "A",
        "Recommended_use": "Main or Supplement",
        "Reason": "Shows representative force decay and renewal."
    },
    {
        "Panel": "B",
        "Recommended_use": "Main",
        "Reason": "Shows clinical renewal lies far below the dynamic window."
    },
    {
        "Panel": "C",
        "Recommended_use": "Main or Supplement",
        "Reason": "Shows dynamic period scaling with tau."
    },
    {
        "Panel": "D",
        "Recommended_use": "Supplement",
        "Reason": "Summarizes force-maintenance properties."
    },
    {
        "Panel": "E",
        "Recommended_use": "Main or Supplement",
        "Reason": "Shows imported Part 2 factor difference between clinical-scale and dynamic regimes."
    },
    {
        "Panel": "F",
        "Recommended_use": "Main",
        "Reason": "Shows period-scale separation."
    },
    {
        "Panel": "G",
        "Recommended_use": "Supplement",
        "Reason": "Shows renewal event schedules without implementation details."
    },
    {
        "Panel": "H",
        "Recommended_use": "Supplement",
        "Reason": "Shows computed order-of-magnitude separation across tau assumptions."
    },
    {
        "Panel": "I",
        "Recommended_use": "Writing aid or graphical abstract draft",
        "Reason": "Shows manuscript logic without implementation details."
    },
    {
        "Panel": "J",
        "Recommended_use": "Supplement",
        "Reason": "Provides period lookup table."
    },
])

figure_recommendation_table.to_csv(os.path.join(OUTDIR, "part5_v3_figure_recommendation_table.csv"), index=False)

# ============================================================
# 9. Figures A-J
# ============================================================

# A. Representative clinical force-decay profiles
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

for profile in df_profiles["Profile"].values:
    sub = df_force[df_force["Profile"] == profile]
    ax.plot(
        sub["Time_days"],
        sub["Normalized_force"],
        linewidth=1.5 if profile != "Ideal static reference" else 1.2,
        color=COLORS.get(profile, "#4A6572"),
        alpha=0.95,
        label=profile
    )

ax.set_xlabel("Time (days)")
ax.set_ylabel("Normalized force")
ax.set_ylim(0, 1.08)
ax.set_title("Clinical force delivery can be represented as force decay and renewal", fontsize=8.7, loc="left")
ax.legend(fontsize=4.7, loc="upper right", ncol=2)
panel_label(ax, "A")
save_fig(fig, "part5_v3_panel_A_clinical_force_decay_profiles.png")

# B. Effective omega*tau of clinical renewal versus Part 2 window
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

plot_ts = df_timescale[df_timescale["Profile"] != "Ideal static reference"].copy()
clinical_rows = plot_ts[plot_ts["Category"] == "Clinical-scale renewal"].copy()
micro_rows = plot_ts[plot_ts["Category"] == "Dynamic stimulation"].copy()

y_labels = list(clinical_rows["Profile"].values) + list(micro_rows["Profile"].values)
profile_to_y = {p: i for i, p in enumerate(y_labels)}

ax.axvspan(PART2_WINDOW_LOW, PART2_WINDOW_HIGH, color="#A67B7B", alpha=0.15, label="Part 2 90% window")

for _, row in clinical_rows.iterrows():
    y = profile_to_y[row["Profile"]]
    ax.scatter(
        row["Effective_omega_tau_at_baseline_tau"],
        y,
        s=55,
        color=COLORS.get(row["Profile"], "#4A6572"),
        edgecolor="white",
        linewidth=0.6,
        zorder=3
    )

for _, row in micro_rows.iterrows():
    y = profile_to_y[row["Profile"]]
    ax.scatter(
        row["Effective_omega_tau_at_baseline_tau"],
        y,
        s=70,
        color=COLORS.get(row["Profile"], "#2d2d2d"),
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        marker="D"
    )

ax.set_xscale("log")
ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=5.8)
ax.set_xlabel(r"Effective $\omega\tau_{\mathrm{ref}}$ at baseline $\tau_{\mathrm{ref}}$")
ax.set_title("Clinical renewal intervals are far below the dynamic window", fontsize=8.7, loc="left")
ax.legend(fontsize=5.5, loc="lower right")
panel_label(ax, "B")
save_fig(fig, "part5_v3_panel_B_clinical_renewal_vs_part2_window.png")

# C. Required dynamic period by tau
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

tau_grid = np.logspace(0, 2, 300)
short_period = period_from_omega_tau(PART2_WINDOW_HIGH, tau_grid)
long_period = period_from_omega_tau(PART2_WINDOW_LOW, tau_grid)
best_period = period_from_omega_tau(PART2_BEST_OMEGA_TAU, tau_grid)
balanced_period = period_from_omega_tau(PART3_BALANCED_OMEGA_TAU, tau_grid)

ax.fill_between(
    tau_grid,
    short_period,
    long_period,
    color="#A67B7B",
    alpha=0.15,
    label="Part 2 90% period window"
)

ax.plot(
    tau_grid,
    best_period,
    color="#4A6572",
    linewidth=2.0,
    label="Part 2 center"
)

ax.plot(
    tau_grid,
    balanced_period,
    color="#2d2d2d",
    linewidth=1.4,
    linestyle=":",
    label="Part 3 balanced"
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Measured or assumed τ (s)")
ax.set_ylabel("Required period (s)")
ax.set_title("Dynamic periods scale with tissue or construct τ", fontsize=8.7, loc="left")
ax.legend(fontsize=5.5, loc="upper left")
panel_label(ax, "C")
save_fig(fig, "part5_v3_panel_C_required_microdynamic_period_by_tau.png")

# D. Force availability metrics
metric_profiles = df_timescale[
    df_timescale["Category"].isin(["Reference", "Clinical-scale renewal"])
].copy()

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

x = np.arange(len(metric_profiles))
width = 0.32

ax.bar(
    x - width / 2,
    metric_profiles["Mean_normalized_force_over_56d"],
    width=width,
    color="#7B9EA6",
    edgecolor="white",
    linewidth=0.5,
    label="Mean force"
)

ax.bar(
    x + width / 2,
    metric_profiles["Min_normalized_force_over_56d"],
    width=width,
    color="#A67B7B",
    edgecolor="white",
    linewidth=0.5,
    label="Minimum force"
)

ax.set_xticks(x)
ax.set_xticklabels(metric_profiles["Profile"], rotation=35, ha="right", fontsize=5.4)
ax.set_ylabel("Normalized force")
ax.set_ylim(0, 1.08)
ax.set_title("Clinical profiles mainly differ in force maintenance", fontsize=8.7, loc="left")
ax.legend(fontsize=5.8, loc="upper right")
panel_label(ax, "D")
save_fig(fig, "part5_v3_panel_D_force_availability_metrics.png")

# E. Part 2 full-model factor at clinical and micro scales
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

plot_ts2 = df_timescale[df_timescale["Profile"] != "Ideal static reference"].copy()
x = np.arange(len(plot_ts2))

bars = ax.bar(
    x,
    plot_ts2["Part2_full_model_factor_at_effective_omega_tau"],
    color=[COLORS.get(p, "#4A6572") for p in plot_ts2["Profile"]],
    edgecolor="white",
    linewidth=0.6,
    width=0.65
)

ax.set_xticks(x)
ax.set_xticklabels(plot_ts2["Profile"], rotation=35, ha="right", fontsize=5.4)
ax.set_ylabel("Imported Part 2 full-model factor")
ax.set_ylim(0, 1.08)
ax.set_title("Part 2 factor distinguishes clinical-scale renewal from micro stimulation", fontsize=8.2, loc="left")

for bar, val in zip(bars, plot_ts2["Part2_full_model_factor_at_effective_omega_tau"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.025,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=5.0,
        color="#2d2d2d"
    )

panel_label(ax, "E")
save_fig(fig, "part5_v3_panel_E_part2_factor_clinical_vs_micro.png")

# F. Period-scale map
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

tau_grid = np.logspace(0, 2, 300)

window_short = period_from_omega_tau(PART2_WINDOW_HIGH, tau_grid)
window_long = period_from_omega_tau(PART2_WINDOW_LOW, tau_grid)

ax.fill_between(
    tau_grid,
    window_short,
    window_long,
    color="#A67B7B",
    alpha=0.15,
    label="Part 2 dynamic window"
)

clinical_intervals = [
    ("Daily", day_to_seconds(1)),
    ("4 days", day_to_seconds(4)),
    ("7 days", day_to_seconds(7)),
    ("14 days", day_to_seconds(14)),
    ("28 days", day_to_seconds(28)),
]

for label, sec in clinical_intervals:
    ax.hlines(
        sec,
        xmin=tau_grid.min(),
        xmax=tau_grid.max(),
        color="#999999",
        linewidth=0.8,
        linestyle="--",
        alpha=0.8
    )
    ax.text(
        tau_grid.max() * 0.93,
        sec * 1.03,
        label,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color="#555555"
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Measured or assumed τ (s)")
ax.set_ylabel("Loading or renewal period (s)")
ax.set_title("Clinical renewal and dynamic loading occupy different time scales", fontsize=8.3, loc="left")
ax.legend(fontsize=5.4, loc="lower right")
panel_label(ax, "F")
save_fig(fig, "part5_v3_panel_F_clinical_micro_period_scale_map.png")

# G. Renewal event schedules over 56 days
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

schedule_profiles = [
    "28-day renewal",
    "14-day renewal",
    "7-day renewal",
    "4-day renewal",
    "1-day renewal",
]

for i, profile_name in enumerate(schedule_profiles):
    profile_row = df_profiles[df_profiles["Profile"] == profile_name].iloc[0]
    interval_days = float(profile_row["Renewal_interval_days"])
    events = np.arange(0, SIM_DAYS + EPS, interval_days)

    ax.hlines(
        i,
        0,
        SIM_DAYS,
        color="#DDDDDD",
        linewidth=0.8
    )

    ax.scatter(
        events,
        np.ones_like(events) * i,
        s=22,
        color=COLORS.get(profile_name, "#4A6572"),
        edgecolor="white",
        linewidth=0.4,
        zorder=3
    )

ax.set_yticks(np.arange(len(schedule_profiles)))
ax.set_yticklabels(schedule_profiles, fontsize=5.8)
ax.set_xlabel("Time (days)")
ax.set_title("Representative renewal schedules occur over days to weeks", fontsize=8.4, loc="left")
ax.set_xlim(-1, SIM_DAYS + 1)
panel_label(ax, "G")
save_fig(fig, "part5_v3_panel_G_clinical_renewal_event_schedules.png")

# H. Computed order-of-magnitude separation across tau assumptions
heatmap_profiles = [
    "28-day renewal",
    "14-day renewal",
    "7-day renewal",
    "4-day renewal",
    "1-day renewal",
]

matrix_records = []

for profile_name in heatmap_profiles:
    sub = df_macro_tau[df_macro_tau["Profile"] == profile_name].copy()
    sub = sub.set_index("Tau_seconds").reindex(TAU_ANCHORS_SECONDS)

    matrix_records.append(sub["Orders_below_part2_window_low"].values)

matrix_sep = np.vstack(matrix_records)

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)

im = ax.imshow(
    matrix_sep,
    cmap=CMAP_DIV,
    aspect="auto",
    vmin=np.nanmin(matrix_sep),
    vmax=np.nanmax(matrix_sep)
)

ax.set_xticks(np.arange(len(TAU_ANCHORS_SECONDS)))
ax.set_xticklabels([f"{x:g}" for x in TAU_ANCHORS_SECONDS], fontsize=5.8)
ax.set_yticks(np.arange(len(heatmap_profiles)))
ax.set_yticklabels(heatmap_profiles, fontsize=5.8)
ax.set_xlabel("Assumed τ (s)")
ax.set_title("Clinical renewal remains orders below the Part 2 window", fontsize=8.4, loc="left")

for i in range(matrix_sep.shape[0]):
    for j in range(matrix_sep.shape[1]):
        v = matrix_sep[i, j]
        if np.isfinite(v):
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=5.1,
                color="white" if v > np.nanmean(matrix_sep) else "#2d2d2d"
            )

cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
cbar.set_label("log10(window low / clinical ωτ)", fontsize=6.2)
cbar.ax.tick_params(labelsize=5.8)
panel_label(ax, "H")
save_fig(fig, "part5_v3_panel_H_order_separation_tau_sensitivity.png")

# I. Manuscript logic bridge
fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)
ax.axis("off")

boxes = [
    (0.05, 0.70, 0.25, 0.18, "Part 1-2\nmechanical foundation +\nfrequency window"),
    (0.38, 0.70, 0.25, 0.18, "Part 3\nobjective-dependent\nwaveform ranking"),
    (0.70, 0.70, 0.25, 0.18, "Part 4\nexperimental protocol\ntranslation"),
    (0.05, 0.25, 0.25, 0.18, "Clinical reality\nforce decay + renewal"),
    (0.38, 0.25, 0.25, 0.18, "Part 5\ntime-scale bridge"),
    (0.70, 0.25, 0.25, 0.18, "Discussion\nexperimental validation\nroadmap"),
]

for x0, y0, w, h, text in boxes:
    rect = plt.Rectangle(
        (x0, y0),
        w,
        h,
        facecolor="#EEF3F4",
        edgecolor="#7B9EA6",
        linewidth=1.0
    )
    ax.add_patch(rect)
    ax.text(
        x0 + w / 2,
        y0 + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=6.3,
        color="#2d2d2d"
    )

arrows = [
    ((0.30, 0.79), (0.38, 0.79)),
    ((0.63, 0.79), (0.70, 0.79)),
    ((0.30, 0.34), (0.38, 0.34)),
    ((0.63, 0.34), (0.70, 0.34)),
    ((0.50, 0.70), (0.50, 0.43)),
]

for start, end in arrows:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color="#4A6572", linewidth=1.2)
    )

ax.set_title("Part 5 connects clinical force decay with the dynamic-loading framework", fontsize=8.2, loc="left")
panel_label(ax, "I")
save_fig(fig, "part5_v3_panel_I_manuscript_logic_bridge.png")

# J. Dynamic period lookup table
lookup_display = df_window_period[[
    "Tau_seconds",
    "Part2_best_period_seconds",
    "Window_short_period_seconds",
    "Window_long_period_seconds",
    "Part3_balanced_period_seconds"
]].copy()

table_data = lookup_display.copy()

for col in table_data.columns:
    table_data[col] = table_data[col].map(lambda x: f"{x:.3g}")

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax)
ax.axis("off")

table = ax.table(
    cellText=table_data.values,
    colLabels=["τ (s)", "P2 center\nperiod", "Window\nshort", "Window\nlong", "P3 balanced\nperiod"],
    loc="center",
    cellLoc="center",
    colLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(5.6)
table.scale(1.0, 1.35)

for _, cell in table.get_celld().items():
    cell.set_edgecolor("#D0D0D0")
    cell.set_linewidth(0.4)

ax.set_title("Dynamic period lookup for experimental interpretation", fontsize=8.4, loc="left")
panel_label(ax, "J")
save_fig(fig, "part5_v3_panel_J_microdynamic_period_lookup_table.png")

# ============================================================
# 10. Combined overview
# ============================================================

def create_combined_overview_600dpi():
    if not PIL_AVAILABLE:
        print("Pillow is not available. Combined overview was skipped. Install pillow if needed: pip install pillow")
        return

    panel_files = [
        "part5_v3_panel_A_clinical_force_decay_profiles.png",
        "part5_v3_panel_B_clinical_renewal_vs_part2_window.png",
        "part5_v3_panel_C_required_microdynamic_period_by_tau.png",
        "part5_v3_panel_D_force_availability_metrics.png",
        "part5_v3_panel_E_part2_factor_clinical_vs_micro.png",
        "part5_v3_panel_F_clinical_micro_period_scale_map.png",
        "part5_v3_panel_G_clinical_renewal_event_schedules.png",
        "part5_v3_panel_H_order_separation_tau_sensitivity.png",
        "part5_v3_panel_I_manuscript_logic_bridge.png",
        "part5_v3_panel_J_microdynamic_period_lookup_table.png",
    ]

    panel_paths = []
    for fn in panel_files:
        candidate = os.path.join(OUTDIR, fn)
        if os.path.exists(candidate):
            panel_paths.append(candidate)
        else:
            print(f"Missing panel for combined overview: {fn}")

    if len(panel_paths) == 0:
        print("No panels were found. Combined overview was not generated.")
        return

    images = []
    for candidate in panel_paths:
        img = Image.open(candidate).convert("RGB")
        images.append((os.path.basename(candidate), img))

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

    out_main = os.path.join(OUTDIR, "part5_v3_combined_overview_600dpi.png")
    canvas.save(out_main, dpi=(600, 600))

    out_alias = os.path.join(OUTDIR, "part5_v3_combined_overview.png")
    canvas.save(out_alias, dpi=(600, 600))

    print(f"Combined 600 dpi overview saved to: {out_main}")
    print(f"Combined 600 dpi overview alias saved to: {out_alias}")
    print(f"Combined overview includes {len(panel_paths)} panels.")

create_combined_overview_600dpi()


# ============================================================
# 12. Summary report
# ============================================================

print("\n" + "=" * 90)
print("Part 5 MANUSCRIPT v3 Manuscript version completed.")
print("=" * 90)
print("Script filename recommendation:")
print("part5_clinical_timescale_bridge.py")
print("")
print("Output folder:")
print(OUTDIR)
print("")
print("Required inputs:")
print(PART2_SUMMARY_FILE)
print(PART2_WINDOW_FILE)
print(PART3_BALANCED_FILE)
print("")
print("Central conclusion:")
print("Clinical-scale force renewal over days/weeks is not equivalent to dynamic PDL-scale stimulation over seconds/minutes.")
print("")
print("Part 2 window:")
print(f"Best omega*tau: {PART2_BEST_OMEGA_TAU:.6g}")
print(f"90% window: {PART2_WINDOW_LOW:.6g} to {PART2_WINDOW_HIGH:.6g}")
print(f"Baseline tau_ref: {BASE_TAU_REF_SECONDS:.6g} s")
print("")
print("Part 3 balanced candidate:")
print(f"Waveform: {PART3_BALANCED_WAVEFORM}")
print(f"Class: {PART3_BALANCED_CLASS}")
print(f"Omega*tau: {PART3_BALANCED_OMEGA_TAU:.6g}")
print("")
print("Macro vs micro time-scale table:")
print(df_timescale[[
    "Profile",
    "Category",
    "Renewal_interval_days",
    "Effective_frequency_Hz",
    "Effective_omega_tau_at_baseline_tau",
    "Orders_below_part2_window_low",
    "Part2_full_model_factor_at_effective_omega_tau",
    "Inside_part2_window",
    "Interpretation"
]].to_string(index=False))
print("")
print("Tau-sensitivity preview:")
print(df_macro_tau.head(25).to_string(index=False))
print("")
print("Required dynamic periods by tau:")
print(df_window_period.to_string(index=False))
print("")
print("Decision table:")
print(decision_table.to_string(index=False))
print("")
print("Neutral scale comparison table:")
print(scale_comparison_table.to_string(index=False))
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
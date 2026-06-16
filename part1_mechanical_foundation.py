# ============================================================
# Part 1: Multi-scale Viscoelastic Mechanical Foundation
# Purpose: demonstrate time-dependent mechanical behavior only.
# Individual panels: 5:3 ratio, 300 dpi PNG.
# Combined overview: existing generated panels stitched at 600 dpi.
# This script demonstrates time-dependent mechanical behavior and does not rank biological outcomes.
# Author: Chen Zong
# ============================================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.integrate import odeint, trapezoid
from scipy.signal import square, sawtooth

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

np.random.seed(42)

# ============================================================
# 0. Output folder and publication style
# ============================================================

OUTDIR = "part1_mechanical_foundation_outputs"
os.makedirs(OUTDIR, exist_ok=True)

DPI = 300
COMBINED_DPI = 600
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

MORANDI = [
    "#7B9EA6",
    "#A67B7B",
    "#8A9E82",
    "#B5A07A",
    "#8E85A6",
    "#A69B7B",
]

BAR_COLORS = [
    "#A8C0C8",
    "#96A8B0",
    "#8A9E98",
    "#A69B85",
    "#B09088",
    "#9E8898",
]

MODEL_COLOR_MAP = {
    "Kelvin-Voigt": "#7B9EA6",
    "Maxwell": "#A67B7B",
    "SLS": "#8A9E82",
    "Burgers": "#B5A07A",
    "Prony": "#8E85A6",
}

WAVEFORM_COLOR_MAP = {
    "Constant": "#A8C0C8",
    "Sine": "#A67B7B",
    "Square": "#8A9E82",
    "Triangle": "#B5A07A",
    "Pulse": "#8E85A6",
    "Asymmetric": "#A69B7B",
}

CMAP_SEQ = mcolors.LinearSegmentedColormap.from_list(
    "morandi_seq",
    ["#C8D8DF", "#7B9EA6", "#4A6572"]
)

CMAP_DIV = mcolors.LinearSegmentedColormap.from_list(
    "morandi_div",
    ["#7B9EA6", "#F0EDE8", "#A67B7B"]
)

CMAP_SCORE = mcolors.LinearSegmentedColormap.from_list(
    "morandi_score",
    ["#F4F1EC", "#C8D8DF", "#7B9EA6", "#4A6572"]
)

# ============================================================
# 1. Frozen material and loading parameters
# ============================================================

PARAMS = {
    "E1": 0.5,
    "E2": 2.0,
    "E3": 1.0,
    "eta1": 10.0,
    "eta2": 50.0,
    "eta_burgers": 30.0,
}

F_mean = 1.0
F_amp = 1.0
freq_demo = 0.1

MODEL_ORDER = ["Kelvin-Voigt", "Maxwell", "SLS", "Burgers", "Prony"]
WAVEFORM_ORDER = ["Constant", "Sine", "Square", "Triangle", "Pulse", "Asymmetric"]

# ============================================================
# 2. Loading waveforms
# ============================================================

def w_constant(t):
    return F_mean * np.ones_like(t)

def w_sine(t, f=freq_demo):
    return F_mean + F_amp * np.sin(2.0 * np.pi * f * t)

def w_square(t, f=freq_demo):
    return F_mean + F_amp * square(2.0 * np.pi * f * t)

def w_triangle(t, f=freq_demo):
    return F_mean + F_amp * sawtooth(2.0 * np.pi * f * t, width=0.5)

def w_pulse(t, f=freq_demo, duty=0.2):
    phase = (2.0 * np.pi * f * t) % (2.0 * np.pi)
    high = phase < (2.0 * np.pi * duty)
    norm = np.sqrt(duty * (1.0 - duty))
    return np.where(
        high,
        F_mean + F_amp * (1.0 - duty) / norm,
        F_mean - F_amp * duty / norm
    )

def w_asymmetric(t, f=freq_demo):
    phase = (2.0 * np.pi * f * t) % (2.0 * np.pi)
    rise_frac = 0.1
    rise_phase = 2.0 * np.pi * rise_frac
    raw = np.where(
        phase < rise_phase,
        phase / rise_phase,
        1.0 - (phase - rise_phase) / (2.0 * np.pi - rise_phase)
    )
    centered = raw - 0.5
    std_raw = np.std(centered)
    if std_raw > 1e-9:
        centered = centered / std_raw * F_amp
    return F_mean + centered

WAVEFORMS = {
    "Constant": w_constant,
    "Sine": w_sine,
    "Square": w_square,
    "Triangle": w_triangle,
    "Pulse": w_pulse,
    "Asymmetric": w_asymmetric,
}

def numerical_derivative(f, t, h=1e-3):
    return (f(t + h) - f(t - h)) / (2.0 * h)

# ============================================================
# 3. Viscoelastic models
# ============================================================

def model_kelvin_voigt(stress_fn, t):
    def rhs(eps, ti):
        return (stress_fn(np.array([ti]))[0] - PARAMS["E1"] * eps) / PARAMS["eta1"]
    return odeint(rhs, 0.0, t).flatten()

def model_maxwell(stress_fn, t):
    eps = np.zeros_like(t)
    dt = t[1] - t[0]
    sigma = stress_fn(t)
    dsigma = np.gradient(sigma, dt)
    for i in range(1, len(t)):
        deps = dsigma[i] / PARAMS["E2"] + sigma[i] / PARAMS["eta1"]
        eps[i] = eps[i - 1] + deps * dt
    return eps

def model_sls(stress_fn, t):
    def rhs(eps, ti):
        sig = stress_fn(np.array([ti]))[0]
        dsig = numerical_derivative(lambda x: stress_fn(np.array([x]))[0], ti)
        return (sig + (PARAMS["eta1"] / PARAMS["E2"]) * dsig - PARAMS["E1"] * eps) / (PARAMS["eta1"] * (1.0 + PARAMS["E1"] / PARAMS["E2"]))
    return odeint(rhs, 0.0, t).flatten()

def model_burgers(stress_fn, t):
    def rhs(state, ti):
        eps_m, eps_kv = state
        sig = stress_fn(np.array([ti]))[0]
        dsig = numerical_derivative(lambda x: stress_fn(np.array([x]))[0], ti)
        deps_m = dsig / PARAMS["E2"] + sig / PARAMS["eta_burgers"]
        deps_kv = (sig - PARAMS["E3"] * eps_kv) / PARAMS["eta2"]
        return [deps_m, deps_kv]
    sol = odeint(rhs, [0.0, 0.0], t)
    return sol[:, 0] + sol[:, 1]

def model_prony(stress_fn, t):
    E_inf = PARAMS["E1"]
    E_modes = [1.5, 0.8, 0.3]
    tau_modes = [0.5, 5.0, 50.0]
    eps = np.zeros_like(t)
    dt = t[1] - t[0]
    sigma = stress_fn(t)
    q = [np.zeros_like(t) for _ in E_modes]
    for i in range(1, len(t)):
        deps_i = (sigma[i] - E_inf * eps[i - 1] - sum(q[k][i - 1] for k in range(len(E_modes)))) / PARAMS["eta1"]
        eps[i] = eps[i - 1] + deps_i * dt
        for k in range(len(E_modes)):
            q[k][i] = q[k][i - 1] + (E_modes[k] * deps_i - q[k][i - 1] / tau_modes[k]) * dt
    return eps

MODELS = {
    "Kelvin-Voigt": model_kelvin_voigt,
    "Maxwell": model_maxwell,
    "SLS": model_sls,
    "Burgers": model_burgers,
    "Prony": model_prony,
}

# ============================================================
# 4. Metrics
# ============================================================

def minmax_normalize(x):
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def robust_normalize(x, lower_q=0.02, upper_q=0.98):
    x = np.asarray(x, dtype=float)
    lo = np.nanquantile(x, lower_q)
    hi = np.nanquantile(x, upper_q)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

def compute_metrics(strain, t, stress):
    dt = t[1] - t[0]
    deps = np.gradient(strain, dt)
    half = len(t) // 2
    threshold = 0.01 * np.max(np.abs(deps) + 1e-9)
    early = np.mean(np.abs(deps[:len(t) // 10]))
    late = np.mean(np.abs(deps[-len(t) // 10:]))
    return {
        "Cumulative strain": trapezoid(strain[half:], t[half:]),
        "Mean strain rate": np.mean(np.abs(deps[half:])),
        "Integrated strain rate": trapezoid(np.abs(deps[half:]), t[half:]),
        "Peak strain": np.max(strain),
        "Energy density": trapezoid(np.abs(stress[half:] * deps[half:]), t[half:]),
        "Fatigue index": trapezoid(stress[half:] ** 2, t[half:]),
        "Effective stim ratio": np.mean(np.abs(deps[half:]) > threshold),
        "Signal decay rate": (early - late) / (early + 1e-9),
        "Early strain-rate": early,
        "Late strain-rate": late,
    }

# ============================================================
# 5. Run all 30 combinations
# ============================================================

t = np.linspace(0.0, 200.0, 10000)
results = []

print("Running 5 models × 6 waveforms = 30 simulations ...")

for m_name, m_func in MODELS.items():
    for w_name, w_func in WAVEFORMS.items():
        stress = w_func(t)
        strain = m_func(w_func, t)
        metrics = compute_metrics(strain, t, stress)
        results.append({
            "Model": m_name,
            "Waveform": w_name,
            "strain": strain,
            "stress": stress,
            **metrics
        })
        print(f"  {m_name:14s} | {w_name:12s} done")

df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("strain", "stress")}
    for r in results
])

df["Mechanical activation proxy"] = robust_normalize(df["Integrated strain rate"])
df["Raw overload proxy"] = robust_normalize(
    0.55 * robust_normalize(df["Peak strain"]) +
    0.30 * robust_normalize(df["Energy density"]) +
    0.15 * robust_normalize(df["Fatigue index"])
)
df["Mechanical-only net score"] = df["Mechanical activation proxy"] * np.maximum(1.0 - df["Raw overload proxy"], 0.0)

df.to_csv(os.path.join(OUTDIR, "part1_metrics.csv"), index=False)

df_summary_model = (
    df.groupby("Model", as_index=False)
    .agg({
        "Integrated strain rate": "mean",
        "Peak strain": "mean",
        "Energy density": "mean",
        "Effective stim ratio": "mean",
        "Signal decay rate": "mean",
        "Mechanical activation proxy": "mean",
        "Raw overload proxy": "mean",
        "Mechanical-only net score": "mean",
    })
)
df_summary_model.to_csv(os.path.join(OUTDIR, "part1_model_summary_mechanical_only.csv"), index=False)

df_summary_waveform = (
    df.groupby("Waveform", as_index=False)
    .agg({
        "Integrated strain rate": "mean",
        "Peak strain": "mean",
        "Energy density": "mean",
        "Effective stim ratio": "mean",
        "Signal decay rate": "mean",
        "Mechanical activation proxy": "mean",
        "Raw overload proxy": "mean",
        "Mechanical-only net score": "mean",
    })
)
df_summary_waveform.to_csv(os.path.join(OUTDIR, "part1_waveform_summary_mechanical_only.csv"), index=False)

interpretation_note_table = pd.DataFrame([
    {
        "Rule": "Part 1 is mechanical-only",
        "Meaning": "Part 1 is allowed to show strong activation by abrupt loading, but it is not used as the primary biological waveform ranking."
    },
    {
        "Rule": "Part 2 adds frequency-window filtering",
        "Meaning": "High-frequency and overly slow regimes are penalized after this point."
    },
    {
        "Rule": "Part 3 performs waveform selection",
        "Meaning": "Equal-energy waveform ranking belongs to Part 3, not Part 1."
    },
    {
        "Rule": "Square/Pulse interpretation",
        "Meaning": "If Square or Pulse appears mechanically strong in Part 1, it means high mechanical activation, not biological optimality."
    },
])
interpretation_note_table.to_csv(os.path.join(OUTDIR, "part1_interpretation_interpretation_notes.csv"), index=False)

# ============================================================
# 6. Plotting helpers
# ============================================================

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

def save_single_panel(fig, filename_base):
    fig.set_size_inches(FIGSIZE[0], FIGSIZE[1], forward=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(
        os.path.join(OUTDIR, f"{filename_base}.png"),
        dpi=DPI,
        facecolor=BG
    )
    plt.close(fig)

def get_result(model_name, waveform_name):
    for r in results:
        if r["Model"] == model_name and r["Waveform"] == waveform_name:
            return r
    raise ValueError(f"Missing result: {model_name}, {waveform_name}")

# ============================================================
# 7A. Panel A: Loading waveforms
# ============================================================

fig, ax_A = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax_A)

t_short = t[:1500]

for name, c in zip(WAVEFORM_ORDER, MORANDI):
    fn = WAVEFORMS[name]
    ax_A.plot(
        t_short,
        fn(t_short),
        color=c,
        label=name,
        linewidth=1.25,
        alpha=0.92
    )

ax_A.set_xlabel("Time (s)")
ax_A.set_ylabel("Applied stress")
ax_A.set_title("Loading waveforms", fontsize=9, loc="left")
ax_A.legend(ncol=2, fontsize=5.8, loc="upper right")
panel_label(ax_A, "A")
save_single_panel(fig, "part1_panel_A_loading_waveforms")

# ============================================================
# 7B. Panel B: Model comparison under sine
# ============================================================

fig, ax_B = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax_B)

for m_name, c in zip(MODEL_ORDER, MORANDI):
    m_func = MODELS[m_name]
    strain_b = m_func(w_sine, t)
    ax_B.plot(
        t[:3000],
        strain_b[:3000],
        color=c,
        label=m_name,
        linewidth=1.25,
        alpha=0.92
    )

ax_B.set_xlabel("Time (s)")
ax_B.set_ylabel("Strain")
ax_B.set_title("Model comparison under sine loading", fontsize=9, loc="left")
ax_B.legend(fontsize=5.8, loc="upper left")
panel_label(ax_B, "B")
save_single_panel(fig, "part1_panel_B_model_comparison_sine")

# ============================================================
# 7C. Panel C: Constant vs sine in SLS with envelope
# ============================================================

fig, ax_C = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax_C)

sls_const = model_sls(w_constant, t)
sls_sine = model_sls(w_sine, t)
zoom_idx = t <= 60.0

ax_C.plot(
    t[zoom_idx],
    sls_const[zoom_idx],
    color="#4A6572",
    linewidth=2.0,
    label="Constant",
    zorder=3
)

ax_C.plot(
    t[zoom_idx],
    sls_sine[zoom_idx],
    color="#A67B7B",
    linewidth=1.0,
    alpha=0.88,
    label="Sine",
    zorder=2
)

ax_C.fill_between(
    t[zoom_idx],
    sls_const[zoom_idx] - 0.55,
    sls_const[zoom_idx] + 0.55,
    color="#A67B7B",
    alpha=0.10,
    label="Envelope",
    zorder=1
)

ax_C.set_xlabel("Time (s)")
ax_C.set_ylabel("Strain")
ax_C.set_title("Relaxation versus sustained activation", fontsize=9, loc="left")
ax_C.legend(fontsize=5.8, loc="lower right")
panel_label(ax_C, "C")
save_single_panel(fig, "part1_panel_C_constant_vs_sine_sls")

# ============================================================
# 7D. Panel D: Strain-rate persistence
# ============================================================

fig, ax_D = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax_D)

dt_local = t[1] - t[0]
deps_const = np.gradient(sls_const, dt_local)
deps_sine = np.gradient(sls_sine, dt_local)

ax_D.plot(
    t,
    np.abs(deps_const) + 1e-12,
    color="#4A6572",
    linewidth=1.5,
    label="Constant"
)

ax_D.plot(
    t,
    np.abs(deps_sine) + 1e-12,
    color="#A67B7B",
    linewidth=0.8,
    alpha=0.75,
    label="Sine"
)

ax_D.set_xlabel("Time (s)")
ax_D.set_ylabel(r"$|d\varepsilon/dt|$")
ax_D.set_yscale("log")
ax_D.set_ylim(1e-6, 1)
ax_D.set_title("Strain-rate signal persistence", fontsize=9, loc="left")
ax_D.legend(fontsize=5.8, loc="lower left")
panel_label(ax_D, "D")
save_single_panel(fig, "part1_panel_D_strain_rate_persistence")

# ============================================================
# 7E. Panel E: Effective stimulation ratio heatmap
# ============================================================

fig, ax_E = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax_E)

heat_data = (
    df.pivot(index="Model", columns="Waveform", values="Effective stim ratio")
    [WAVEFORM_ORDER]
    .reindex(MODEL_ORDER)
)

im = ax_E.imshow(
    heat_data.values,
    cmap=CMAP_SEQ,
    aspect="auto",
    vmin=0,
    vmax=1
)

ax_E.set_xticks(range(len(WAVEFORM_ORDER)))
ax_E.set_xticklabels(WAVEFORM_ORDER, rotation=30, ha="right", fontsize=5.8)
ax_E.set_yticks(range(len(MODEL_ORDER)))
ax_E.set_yticklabels(MODEL_ORDER, fontsize=5.8)
ax_E.set_title("Effective stimulation ratio", fontsize=9, loc="left")

for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        v = heat_data.values[i, j]
        ax_E.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            color="white" if v > 0.6 else "#2d2d2d",
            fontsize=5.8
        )

cbar = plt.colorbar(im, ax=ax_E, shrink=0.78, pad=0.02)
cbar.set_label("Stim ratio", fontsize=6.5)
cbar.ax.tick_params(labelsize=5.8)
panel_label(ax_E, "E")
save_single_panel(fig, "part1_panel_E_effective_stimulation_ratio_heatmap")

# ============================================================
# 7F. Panel F: Integrated strain-rate signal in SLS
# ============================================================

fig, ax_F = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax_F)

sls_rows = (
    df[df["Model"] == "SLS"]
    .set_index("Waveform")
    .reindex(WAVEFORM_ORDER)
)

bars = ax_F.bar(
    range(len(sls_rows)),
    sls_rows["Integrated strain rate"],
    color=BAR_COLORS,
    edgecolor="white",
    linewidth=0.6,
    width=0.65
)

ax_F.set_xticks(range(len(sls_rows)))
ax_F.set_xticklabels(sls_rows.index, rotation=30, ha="right", fontsize=5.8)
ax_F.set_ylabel(r"$\int |d\varepsilon/dt|\,dt$")
ax_F.set_title("Cumulative strain-rate signal", fontsize=9, loc="left")

ymax = float(np.nanmax(sls_rows["Integrated strain rate"]))
ax_F.set_ylim(0, ymax * 1.22)

for bar, val in zip(bars, sls_rows["Integrated strain rate"]):
    ax_F.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + ymax * 0.025,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#2d2d2d"
    )

panel_label(ax_F, "F")
save_single_panel(fig, "part1_panel_F_integrated_strain_rate_sls")

# ============================================================
# 7G. Panel G: Normalized mechanical-only activation-overload map
# ============================================================

fig, ax_G = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax_G)

for m_name in MODEL_ORDER:
    sub = df[df["Model"] == m_name]
    ax_G.scatter(
        sub["Raw overload proxy"],
        sub["Mechanical activation proxy"],
        color=MODEL_COLOR_MAP[m_name],
        label=m_name,
        s=46,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.92
    )

for _, row in df.iterrows():
    if row["Mechanical-only net score"] >= df["Mechanical-only net score"].quantile(0.80):
        label = f"{row['Model'].split('-')[0][:2]}-{row['Waveform'][:3]}"
        ax_G.text(
            row["Raw overload proxy"] + 0.012,
            row["Mechanical activation proxy"] + 0.012,
            label,
            fontsize=5.0,
            color="#2d2d2d"
        )

ax_G.set_xlabel("Normalized raw mechanical-overload proxy")
ax_G.set_ylabel("Normalized mechanical-activation proxy")
ax_G.set_xlim(-0.04, 1.04)
ax_G.set_ylim(-0.04, 1.04)
ax_G.set_title("Mechanical-only activation-overload map", fontsize=9, loc="left")
ax_G.legend(fontsize=5.2, loc="lower right", ncol=2)
panel_label(ax_G, "G")
save_single_panel(fig, "part1_panel_G_mechanical_only_activation_overload_map")

# ============================================================
# 7H. Panel H: Signal decay rate heatmap
# ============================================================

fig, ax_H = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax_H)

decay_pivot = (
    df.pivot(index="Model", columns="Waveform", values="Signal decay rate")
    [WAVEFORM_ORDER]
    .reindex(MODEL_ORDER)
)

im2 = ax_H.imshow(
    decay_pivot.values,
    cmap=CMAP_DIV,
    aspect="auto",
    vmin=-1,
    vmax=1
)

ax_H.set_xticks(range(len(WAVEFORM_ORDER)))
ax_H.set_xticklabels(WAVEFORM_ORDER, rotation=30, ha="right", fontsize=5.8)
ax_H.set_yticks(range(len(MODEL_ORDER)))
ax_H.set_yticklabels(MODEL_ORDER, fontsize=5.8)
ax_H.set_title("Signal decay rate", fontsize=9, loc="left")

for i in range(decay_pivot.shape[0]):
    for j in range(decay_pivot.shape[1]):
        v = decay_pivot.values[i, j]
        ax_H.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            color="white" if abs(v) > 0.55 else "#2d2d2d",
            fontsize=5.8
        )

cbar2 = plt.colorbar(im2, ax=ax_H, shrink=0.78, pad=0.02)
cbar2.set_label("Decay rate", fontsize=6.5)
cbar2.ax.tick_params(labelsize=5.8)
panel_label(ax_H, "H")
save_single_panel(fig, "part1_panel_H_signal_decay_rate_heatmap")

# ============================================================
# 7I. Panel I: Original raw efficacy-risk trade-off as supplement
# ============================================================

fig, ax_I = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax_I)

for m_name in MODEL_ORDER:
    sub = df[df["Model"] == m_name]
    ax_I.scatter(
        sub["Peak strain"],
        sub["Integrated strain rate"],
        color=MODEL_COLOR_MAP[m_name],
        label=m_name,
        s=38,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.90
    )

ax_I.set_xlabel("Peak strain")
ax_I.set_ylabel("Integrated strain-rate")
ax_I.set_xscale("log")
ax_I.set_yscale("log")
ax_I.set_title("Raw efficacy-risk trade-off", fontsize=9, loc="left")
ax_I.legend(fontsize=5.5, loc="lower right")
panel_label(ax_I, "I")
save_single_panel(fig, "part1_panel_I_raw_efficacy_risk_tradeoff")

# ============================================================
# 7J. Panel J: Mechanical-only score matrix
# ============================================================

fig, ax_J = plt.subplots(figsize=FIGSIZE, facecolor=BG)
style_ax(ax_J)

score_data = (
    df.pivot(index="Model", columns="Waveform", values="Mechanical-only net score")
    [WAVEFORM_ORDER]
    .reindex(MODEL_ORDER)
)

im3 = ax_J.imshow(
    score_data.values,
    cmap=CMAP_SCORE,
    aspect="auto",
    vmin=0,
    vmax=max(1.0, float(np.nanmax(score_data.values)))
)

ax_J.set_xticks(range(len(WAVEFORM_ORDER)))
ax_J.set_xticklabels(WAVEFORM_ORDER, rotation=30, ha="right", fontsize=5.8)
ax_J.set_yticks(range(len(MODEL_ORDER)))
ax_J.set_yticklabels(MODEL_ORDER, fontsize=5.8)
ax_J.set_title("Mechanical-only score matrix", fontsize=9, loc="left")

for i in range(score_data.shape[0]):
    for j in range(score_data.shape[1]):
        v = score_data.values[i, j]
        ax_J.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            color="white" if v > 0.60 else "#2d2d2d",
            fontsize=5.8
        )

cbar3 = plt.colorbar(im3, ax=ax_J, shrink=0.78, pad=0.02)
cbar3.set_label("Mechanical-only score", fontsize=6.5)
cbar3.ax.tick_params(labelsize=5.8)
panel_label(ax_J, "J")
save_single_panel(fig, "part1_panel_J_mechanical_only_score_matrix")

# ============================================================
# 8. Combined 600 dpi overview from selected panels
# ============================================================

def create_combined_overview_600dpi():
    if not PIL_AVAILABLE:
        print("Pillow is not available. Combined overview was skipped. Install pillow if needed: pip install pillow")
        return

    panel_files = [
        "part1_panel_A_loading_waveforms.png",
        "part1_panel_B_model_comparison_sine.png",
        "part1_panel_C_constant_vs_sine_sls.png",
        "part1_panel_D_strain_rate_persistence.png",
        "part1_panel_E_effective_stimulation_ratio_heatmap.png",
        "part1_panel_F_integrated_strain_rate_sls.png",
        "part1_panel_G_mechanical_only_activation_overload_map.png",
        "part1_panel_H_signal_decay_rate_heatmap.png",
        "part1_panel_I_raw_efficacy_risk_tradeoff.png",
        "part1_panel_J_mechanical_only_score_matrix.png",
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

    out_path = os.path.join(OUTDIR, "part1_combined_overview_600dpi.png")
    canvas.save(out_path, dpi=(600, 600))

    out_path_alias = os.path.join(OUTDIR, "part1_combined_overview.png")
    canvas.save(out_path_alias, dpi=(600, 600))

    print(f"Combined 600 dpi overview saved to: {out_path}")
    print(f"Combined 600 dpi overview alias saved to: {out_path_alias}")
    print(f"Combined overview includes {len(panel_paths)} panels.")

create_combined_overview_600dpi()

# ============================================================
# 9. Summary report
# ============================================================

print("\n" + "=" * 80)
print("Part 1 mechanical-foundation analysis completed.")
print("All outputs were saved to:")
print(OUTDIR)
print("Expected individual PNG size: 1500 × 900 px at 300 dpi.")
print("Combined overview: 600 dpi.")
print("Important: Part 1 is mechanical-only and is not used as the primary biological waveform ranking.")
print(f"Total simulations: {len(results)}")
print("=" * 80)
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
print("")
print("Mechanical-only waveform summary:")
print(df_summary_waveform.to_string(index=False))
print("=" * 80)

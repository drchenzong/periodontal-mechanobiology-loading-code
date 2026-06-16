# ============================================================
# part7_figure6AB_rheology_ONLY_v7.py
#
# Figure 6A-B only:
#   6A Non-Newtonian flow behaviour
#   6B Elastic-dominant viscoelastic gel
#
# No PDF output.
# No SEM / ALP / ARS processing.
# No panel letters. Add A/B/C/D later in Illustrator.
#
# Run this script from the repository root.
# Required data folder: data/rheology/
# Outputs: outputs/figure6AB_rheology/
# ============================================================

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from scipy.interpolate import PchipInterpolator
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ============================================================
# 0. Paths
# ============================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "rheology"
OUTDIR = SCRIPT_DIR / "outputs" / "figure6AB_rheology"
OUTDIR.mkdir(parents=True, exist_ok=True)

candidate_files = []
candidate_files += list(DATA_DIR.glob("LLY.txt"))
candidate_files += list(DATA_DIR.glob("rheometry*.txt"))
candidate_files += list(DATA_DIR.glob("*2024-07-29*.txt"))

if not candidate_files:
    raise FileNotFoundError(
        f"Cannot find rheology txt file in: {DATA_DIR}\n"
        "Please provide a rheology text file in data/rheology/."
    )

RHEO_FILE = candidate_files[0]
print(f"Using rheology file: {RHEO_FILE.name}")

# ============================================================
# 1. Morandi palette
# ============================================================

C_BLUE_DARK = "#4A6870"
C_RED = "#A97975"
C_BG = "#FAFAFA"
C_DARK = "#222222"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "axes.facecolor": C_BG,
    "figure.facecolor": C_BG,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

DPI = 600

# ============================================================
# 2. Read rheology text
# ============================================================

def read_text_safely(path):
    raw = path.read_bytes()

    for enc in ["utf-16-le", "utf-16", "utf-8-sig", "utf-8", "latin1"]:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass

    return raw.decode("utf-8", errors="replace")


text = read_text_safely(RHEO_FILE)

# ============================================================
# 3. Parse rheology data
# ============================================================

def parse_rheology(text):
    results = {}

    test_blocks = re.split(r"Test:\s*", text)[1:]

    for block in test_blocks:
        tid = block.split("\n")[0].strip().split("\r")[0].strip()
        if not tid:
            continue

        flow_data = []

        fc = re.search(
            r"Flow curve 1.*?Interval and data points:\s*1\s+(\d+)(.*?)"
            r"(?=Interval and data points:|Result:|Gel|Test:|$)",
            block,
            re.DOTALL
        )

        if fc:
            n_pts = int(fc.group(1))
            block_text = fc.group(2)

            for line in block_text.splitlines():
                nums = []
                for p in re.split(r"[\t ]+", line.strip()):
                    try:
                        nums.append(float(p.strip().replace(",", "")))
                    except Exception:
                        pass

                if len(nums) >= 4:
                    pt = int(nums[0])
                    sr = nums[1]
                    visc = nums[3]

                    if 1 <= pt <= n_pts and sr > 0 and visc > 0:
                        flow_data.append((sr, visc))

        gel_data = []

        gm = re.search(
            r"Gel 1.*?Interval and data points:\s*2\s+(\d+)(.*?)"
            r"(?=Result:|Test:|$)",
            block,
            re.DOTALL
        )

        if gm:
            n_pts = int(gm.group(1))
            block_text = gm.group(2)

            for line in block_text.splitlines():
                nums = []
                for p in re.split(r"[\t ]+", line.strip()):
                    try:
                        nums.append(float(p.strip().replace(",", "")))
                    except Exception:
                        pass

                if len(nums) >= 3:
                    try:
                        pt = int(nums[0])
                        gp = nums[-2]
                        gpp = nums[-1]

                        if 1 <= pt <= n_pts and gp > 0 and gpp >= 0:
                            gel_data.append((pt, gp, gpp))
                    except Exception:
                        pass

        results[tid] = {
            "flow": flow_data,
            "gel": gel_data,
        }

    return results


rheo = parse_rheology(text)

if "4" not in rheo:
    raise ValueError(
        f"Test 4 not found in {RHEO_FILE.name}. Available tests: {list(rheo.keys())}"
    )

d4 = rheo["4"]

flow_arr = np.array(d4["flow"], dtype=float)
gel_arr = np.array(d4["gel"], dtype=float)

if flow_arr.size == 0:
    raise ValueError("No flow-curve data parsed for Test 4.")

if gel_arr.size == 0:
    raise ValueError("No gelation/time-sweep data parsed for Test 4.")

sr_all = flow_arr[:, 0]
visc_all = flow_arr[:, 1]

gel_pts = gel_arr[:, 0]
gp_all = gel_arr[:, 1]
gpp_all = gel_arr[:, 2]

t_sec = gel_pts - 1

sort_idx = np.argsort(sr_all)
sr_all = sr_all[sort_idx]
visc_all = visc_all[sort_idx]

# ============================================================
# 4. Summary statistics
# ============================================================

gp_plat = gp_all[-100:]
gpp_plat = gpp_all[-100:]

gp_mean = float(np.mean(gp_plat))
gpp_mean = float(np.mean(gpp_plat))
tan_delta = gpp_mean / gp_mean

omega_gel = 10.0
tau_app_s = tan_delta / omega_gel
tau_app_ms = tau_app_s * 1000

log_sr = np.log10(sr_all)
log_visc = np.log10(visc_all)

coef = np.polyfit(log_sr, log_visc, 1)
slope = coef[0]
n_index = slope + 1

sr_smooth = np.linspace(sr_all.min(), sr_all.max(), 700)

if SCIPY_AVAILABLE:
    smooth_func = PchipInterpolator(sr_all, visc_all)
    visc_smooth = smooth_func(sr_smooth)
else:
    visc_smooth = np.interp(sr_smooth, sr_all, visc_all)

print("\nParsed 4 mg/mL rheology:")
print(f"Flow curve: {sr_all.min():.0f}–{sr_all.max():.0f} s^-1")
print(f"Viscosity: {visc_all[0]:.1f} → {visc_all[-1]:.1f} mPa·s")
print(f"Fold decrease: {visc_all[0] / visc_all[-1]:.1f}")
print(f"Power-law index n = {n_index:.3f}")
print(f"G′ plateau = {gp_mean:.1f} Pa")
print(f"G″ plateau = {gpp_mean:.1f} Pa")
print(f"tanδ = {tan_delta:.3f}")
print(f"tau_app = {tau_app_ms:.1f} ms at omega = 10 rad/s")

# ============================================================
# 5. Helper
# ============================================================

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(False)

# ============================================================
# 6. Figure 6A
# ============================================================

def make_figure_6A():
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    fig.patch.set_facecolor(C_BG)

    ax.plot(
        sr_smooth,
        visc_smooth,
        color=C_BLUE_DARK,
        lw=2.25,
        alpha=0.98,
        label="Smoothed curve",
        zorder=2
    )

    ax.scatter(
        sr_all,
        visc_all,
        s=22,
        color=C_RED,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
        label=f"Raw data, $n$ = {n_index:.2f}"
    )

    ax.set_xlim(0, 102)
    ax.set_ylim(0, 1000)

    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 200, 400, 600, 800, 1000])

    ax.set_xlabel(r"Shear rate (s$^{-1}$)")
    ax.set_ylabel(r"Viscosity (mPa$\cdot$s)")
    ax.set_title("Non-Newtonian flow behaviour", pad=7)

    drop_fold = visc_all[0] / visc_all[-1]

    annotation = (
        f"{visc_all[0]:.0f} → {visc_all[-1]:.1f} mPa·s\n"
        f"{drop_fold:.1f}-fold decrease"
    )

    ax.text(
        0.96,
        0.70,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color=C_DARK,
        bbox=dict(
            facecolor=C_BG,
            edgecolor="none",
            alpha=0.88,
            pad=2.5
        )
    )

    ax.legend(
        frameon=False,
        loc="upper right",
        handlelength=2.0
    )

    clean_axes(ax)
    fig.tight_layout(pad=0.75)

    out = OUTDIR / "fig6A_flow_curve_600dpi.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    print(f"Saved: {out}")
    return out

# ============================================================
# 7. Figure 6B
# ============================================================

def make_figure_6B():
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    fig.patch.set_facecolor(C_BG)

    ax.plot(
        t_sec,
        gp_all,
        color=C_BLUE_DARK,
        lw=2.05,
        label=r"$G'$ storage"
    )

    ax.plot(
        t_sec,
        gpp_all,
        color=C_RED,
        lw=1.9,
        ls="--",
        label=r"$G''$ loss"
    )

    ax.axhline(
        gp_mean,
        color=C_BLUE_DARK,
        lw=0.9,
        ls=":",
        alpha=0.45
    )

    ax.axhline(
        gpp_mean,
        color=C_RED,
        lw=0.9,
        ls=":",
        alpha=0.45
    )

    ax.set_xlim(-3, 363)
    ax.set_ylim(-5, max(gp_all) * 1.14)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])

    ax.set_xlabel("Time at 37°C (s)")
    ax.set_ylabel("Modulus (Pa)")
    ax.set_title("Elastic-dominant viscoelastic gel", pad=7)

    stats_txt = (
        f"$G'$ = {gp_mean:.0f} Pa\n"
        f"$G''$ = {gpp_mean:.1f} Pa\n"
        f"tanδ = {tan_delta:.3f}\n"
        f"τ$_{{app}}$ ≈ {tau_app_ms:.1f} ms"
    )

    ax.text(
        0.97,
        0.57,
        stats_txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=C_DARK,
        bbox=dict(
            facecolor=C_BG,
            edgecolor="none",
            alpha=0.90,
            pad=2.5
        )
    )

    ax.legend(
        frameon=False,
        loc="upper left",
        handlelength=2.0
    )

    clean_axes(ax)
    fig.tight_layout(pad=0.75)

    out = OUTDIR / "fig6B_gel_time_sweep_600dpi.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    print(f"Saved: {out}")
    return out

# ============================================================
# 8. Save concise legend txt
# ============================================================

def save_figure6_legend():
    legend_path = OUTDIR / "Figure_6_legend.txt"

    legend = f"""Figure 6. Experimental anchoring of the dimensionless loading framework in a biomimetic collagen-PDLSC hydrogel.

(A) Flow behaviour of the 4 mg/mL hydrogel showing marked shear-thinning, with viscosity decreasing from {visc_all[0]:.0f} to {visc_all[-1]:.1f} mPa·s and a power-law flow index of n = {n_index:.2f}. (B) Oscillatory time-sweep rheology showing elastic-dominant gel behaviour, with plateau G′ = {gp_mean:.0f} Pa, G″ = {gpp_mean:.1f} Pa, tanδ = {tan_delta:.3f}, and τapp ≈ {tau_app_ms:.1f} ms estimated from tanδ/ω at 10 rad/s. (C) Representative SEM images of the hydrogel and native PDL microarchitecture. (D) Representative ALP and Alizarin Red S staining showing PDLSC osteogenic activity and mineral deposition within the hydrogel.
"""

    with open(legend_path, "w", encoding="utf-8") as f:
        f.write(legend)

    print(f"Saved: {legend_path}")

# ============================================================
# 9. Save summary CSV
# ============================================================

def save_summary():
    summary_path = OUTDIR / "fig6AB_rheology_summary.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"viscosity_first_mPa_s,{visc_all[0]:.6g}\n")
        f.write(f"viscosity_last_mPa_s,{visc_all[-1]:.6g}\n")
        f.write(f"viscosity_fold_decrease,{visc_all[0] / visc_all[-1]:.6g}\n")
        f.write(f"power_law_n,{n_index:.6g}\n")
        f.write(f"G_storage_plateau_Pa,{gp_mean:.6g}\n")
        f.write(f"G_loss_plateau_Pa,{gpp_mean:.6g}\n")
        f.write(f"tan_delta,{tan_delta:.6g}\n")
        f.write(f"tau_app_ms,{tau_app_ms:.6g}\n")

    print(f"Saved: {summary_path}")

# ============================================================
# 10. Run
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Figure 6A-B only: updated titles")
    print("=" * 70)

    make_figure_6A()
    make_figure_6B()
    save_figure6_legend()
    save_summary()

    print("\nDone.")
    print(f"Outputs are in: {OUTDIR}")
    print("=" * 70)
    


# ============================================================
# 11. YAP/TAZ
# Updated version: larger fonts, wider layout, sparse y-ticks
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Raw data
# -----------------------------
taz_below_raw  = [0.924, 0.715, 1.064]
taz_within_raw = [1.626, 1.348, 1.491]
yap_below_raw  = [0.682, 0.581, 0.982]
yap_within_raw = [0.897, 0.789, 1.061]

taz_below_mean  = np.mean(taz_below_raw)
taz_below_sd    = np.std(taz_below_raw, ddof=1)
taz_within_mean = np.mean(taz_within_raw)
taz_within_sd   = np.std(taz_within_raw, ddof=1)

yap_below_mean  = np.mean(yap_below_raw)
yap_below_sd    = np.std(yap_below_raw, ddof=1)
yap_within_mean = np.mean(yap_within_raw)
yap_within_sd   = np.std(yap_within_raw, ddof=1)

# -----------------------------
# Style
# -----------------------------
C_BELOW  = '#b07070'   # Morandi red
C_WITHIN = '#4a8c8c'   # Morandi blue-green

FONT = 'Arial'

# Enlarged fonts
FS_YLABEL = 12.5
FS_TICK   = 11.5
FS_TITLE  = 13.5
FS_SIG    = 16

# Wider figure, same height
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6), dpi=300)
fig.patch.set_facecolor('white')


def draw_panel(ax, below_raw, within_raw,
               below_mean, below_sd, within_mean, within_sd,
               p_val, ylabel, title,
               ylim_top, yticks):

    x = np.array([0, 1])
    means  = [below_mean, within_mean]
    sds    = [below_sd, within_sd]
    colors = [C_BELOW, C_WITHIN]
    raws   = [below_raw, within_raw]

    # Bars
    ax.bar(
        x, means, width=0.52, color=colors,
        edgecolor='white', linewidth=0.9, zorder=2
    )

    # Error bars
    ax.errorbar(
        x, means, yerr=sds, fmt='none',
        ecolor='#444444', elinewidth=1.2,
        capsize=4.5, capthick=1.2, zorder=4
    )

    # Raw data points
    np.random.seed(42)
    for xi, rv in zip(x, raws):
        jitter = np.random.uniform(-0.07, 0.07, len(rv))
        ax.scatter(
            xi + jitter, rv, s=26,
            color='#222222', linewidths=0,
            alpha=0.88, zorder=5
        )

    # Significance bracket position based on panel scale
    y_sig = ylim_top * 0.86
    y_top = ylim_top * 0.91

    ax.plot(
        [x[0], x[0], x[1], x[1]],
        [y_sig, y_top, y_top, y_sig],
        color='#333333', lw=1.0, zorder=6
    )

    if p_val < 0.05:
        ax.text(
            0.5, ylim_top * 0.935, '*',
            ha='center', va='bottom',
            fontsize=FS_SIG, fontname=FONT, color='#222222'
        )
    else:
        ax.text(
            0.5, ylim_top * 0.935, 'ns',
            ha='center', va='bottom',
            fontsize=FS_TICK, fontname=FONT, color='#444444'
        )

    # Axes formatting
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, ylim_top)
    ax.set_xticks(x)
    ax.set_xticklabels(['1 Hz', '30 Hz'], fontsize=FS_TICK, fontname=FONT)
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [f'{v:.1f}' for v in yticks],
        fontsize=FS_TICK, fontname=FONT
    )

    ax.set_ylabel(ylabel, fontsize=FS_YLABEL, fontname=FONT, labelpad=6)
    ax.set_title(title, fontsize=FS_TITLE, fontname=FONT, pad=7)  # not bold

    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=3.5, width=0.8, color='#444444')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.85)
    ax.spines['bottom'].set_linewidth(0.85)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')


# -----------------------------
# Left panel: TAZ
# -----------------------------
draw_panel(
    axes[0],
    taz_below_raw, taz_within_raw,
    taz_below_mean, taz_below_sd,
    taz_within_mean, taz_within_sd,
    p_val=0.010,
    ylabel='TAZ nuclear/perinuclear ratio',
    title='TAZ',
    ylim_top=2.0,
    yticks=np.arange(0.0, 2.0 + 0.001, 0.5)
)

# -----------------------------
# Right panel: YAP
# -----------------------------
draw_panel(
    axes[1],
    yap_below_raw, yap_within_raw,
    yap_below_mean, yap_below_sd,
    yap_within_mean, yap_within_sd,
    p_val=0.310,
    ylabel='YAP nuclear/perinuclear ratio',
    title='YAP',
    ylim_top=1.2,
    yticks=np.arange(0.0, 1.2 + 0.001, 0.3)
)

# Layout
fig.subplots_adjust(
    left=0.12,
    right=0.985,
    bottom=0.16,
    top=0.90,
    wspace=0.42
)

# Save PNG only
fig.savefig(
    'Figure_6G_v4.png',
    dpi=600,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()
print('Done: Figure_6G_v4.png')
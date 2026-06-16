#============================================================
# Figure assembly script
#
# Assemble selected single-panel outputs into manuscript figures.
#
# Figure assembly workflow.
# Run from the repository root after completing Part 1-5 analyses.
#
# Run from the repository root after completing Part 1-5 analyses:
# python generate_publication_figures.py
#
# Author: Chen Zong
# ============================================================

import re
import math
import subprocess
import time
from pathlib import Path

import pandas as pd

try:
    from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# ============================================================
# 0. Base paths
# ============================================================

BASE_DIR = Path.cwd()
RUN_STARTED_AT = time.time()

OUTDIR = BASE_DIR / "figure_assembly_outputs"
PATCHED_SCRIPT_DIR = OUTDIR / "patched_source_scripts"
SOURCE_OUTPUT_DIR = OUTDIR / "source_regenerated_outputs"
REFINED_PANEL_DIR = OUTDIR / "refined_single_panels"
MAIN_FIGURE_DIR = OUTDIR / "main_figures"
SUPPLEMENTARY_FIGURE_DIR = OUTDIR / "supplementary_figures_S1_to_S8"
QC_FULL_SUPPLEMENTARY_DIR = OUTDIR / "qc_full_supplementary_overviews"
MANIFEST_DIR = OUTDIR / "manifests"
LEGEND_DIR = OUTDIR / "figure_legends"

for d in [
    OUTDIR,
    PATCHED_SCRIPT_DIR,
    SOURCE_OUTPUT_DIR,
    REFINED_PANEL_DIR,
    MAIN_FIGURE_DIR,
    SUPPLEMENTARY_FIGURE_DIR,
    QC_FULL_SUPPLEMENTARY_DIR,
    MANIFEST_DIR,
    LEGEND_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Global publication style
# ============================================================

STYLE = {
    "fig_width_in": 9.0,
    "fig_height_in": 5.4,
    "dpi": 600,

    "single_panel_width_px": 5400,
    "single_panel_height_px": 3240,
    "single_panel_dpi": 600,

    "main_composite_dpi": 600,
    "main_gap_px": 150,
    "main_outer_pad_px": 190,
    "main_border_px": 1,

    "supp_composite_dpi": 600,
    "supp_gap_px": 140,
    "supp_outer_pad_px": 180,
    "supp_border_px": 1,

    "bg_rgb": (250, 250, 250),
    "paper_rgb": (255, 255, 255),
    "border_rgb": (210, 210, 210),

    "base_font_size": 16,
    "title_font_size": 18,
    "axis_font_size": 17,
    "tick_font_size": 14,
    "legend_font_size": 13,
    "table_font_size": 12,

    # Composite-level labels.
    # These are drawn only after panels are assembled.
    "composite_panel_label_size": 200,
    "composite_panel_label_x": 34,
    "composite_panel_label_y": 24,

    "axis_linewidth": 1.25,
    "major_tick_width": 1.05,
    "line_width_default": 2.3,

    "trim_output": True,
    "trim_tolerance": 10,
    "safe_margin_px": 170,

    "default_x_tick_rotation": 35,
}


# ============================================================
# 2. Morandi palette
# ============================================================

COLOR_REWRITE = {
    "#A8C0C8": "#C9D6D9",
    "#96A8B0": "#A9BDC2",
    "#7B9EA6": "#5F818A",
    "#4A6572": "#3E5963",
    "#8A9E82": "#788E70",
    "#B5A07A": "#A9946C",
    "#A69B7B": "#9A8C6A",
    "#8E85A6": "#7E7595",
    "#A67B7B": "#9D6F70",
    "#B09088": "#A77C76",
    "#9E8898": "#927D8E",
    "#D9E3E5": "#D7E0E2",
    "#C8D8DF": "#BFD1D7",
    "#EEF3F4": "#EEF2F2",
    "#F4F1EC": "#F3EFEA",
    "#F0EDE8": "#F0ECE6",
    "#777777": "#707070",
    "#999999": "#8A8A8A",
    "#D0D0D0": "#CFCFCF",
    "#DDDDDD": "#D9D9D9",
}

FORCED_PROP_CYCLE = [
    "#5F818A",
    "#A9946C",
    "#788E70",
    "#9D6F70",
    "#7E7595",
    "#9A8C6A",
    "#A77C76",
    "#A9BDC2",
]


# ============================================================
# 3. Source scripts and output folders
# ============================================================

SOURCE_SCRIPT_ALIASES = {
    "P1": ["part1_mechanical_foundation.py"],
    "P2": ["part2_frequency_window.py"],
    "P3": ["part3_waveform_screening.py"],
    "P4": ["part4_protocol_translation.py"],
    "P5": ["part5_clinical_timescale_bridge.py"],
}

ORIGINAL_OUTPUT_FOLDERS = {
    "P1": "part1_mechanical_foundation_outputs",
    "P2": "part2_frequency_window_outputs",
    "P3": "part3_waveform_screening_outputs",
    "P4": "part4_protocol_translation_outputs",
    "P5": "part5_clinical_timescale_bridge_outputs",
}

PATCHED_OUTPUT_FOLDERS = {
    "P1": str((SOURCE_OUTPUT_DIR / "P1_mechanical_foundation").as_posix()),
    "P2": str((SOURCE_OUTPUT_DIR / "P2_frequency_window").as_posix()),
    "P3": str((SOURCE_OUTPUT_DIR / "P3_waveform_optimization").as_posix()),
    "P4": str((SOURCE_OUTPUT_DIR / "P4_protocol_translation").as_posix()),
    "P5": str((SOURCE_OUTPUT_DIR / "P5_clinical_bridge").as_posix()),
}

PANEL_FILES = {
    "P1": {
        "A": "part1_panel_A_loading_waveforms.png",
        "B": "part1_panel_B_model_comparison_sine.png",
        "C": "part1_panel_C_constant_vs_sine_sls.png",
        "D": "part1_panel_D_strain_rate_persistence.png",
        "E": "part1_panel_E_effective_stimulation_ratio_heatmap.png",
        "F": "part1_panel_F_integrated_strain_rate_sls.png",
        "G": "part1_panel_G_mechanical_only_activation_overload_map.png",
        "H": "part1_panel_H_signal_decay_rate_heatmap.png",
        "I": "part1_panel_I_raw_efficacy_risk_tradeoff.png",
        "J": "part1_panel_J_mechanical_only_score_matrix.png",
    },
    "P2": {
        "A": "part2_v3_panel_A_frequency_response_components.png",
        "B": "part2_v3_panel_B_candidate_frequency_window.png",
        "C": "part2_v3_panel_C_ablation_window_formation.png",
        "D": "part2_v3_panel_D_cellular_cutoff_sensitivity.png",
        "E": "part2_v3_panel_E_mean_load_design_space.png",
        "F": "part2_v3_panel_F_monte_carlo_best_omega_tau.png",
        "G": "part2_v3_panel_G_absolute_frequency_scaling.png",
        "H": "part2_v3_panel_H_phase_dissipation_high_frequency.png",
        "I": "part2_v3_panel_I_cutoff_sensitivity_summary.png",
        "J": "part2_v3_panel_J_decision_summary.png",
    },
    "P3": {
        "A": "part3_v4_panel_A_equal_RMS_waveform_candidates.png",
        "B": "part3_v4_panel_B_hard_part2_window_constraint.png",
        "C": "part3_v4_panel_C_global_vs_window_efficiency.png",
        "D": "part3_v4_panel_D_top_hard_window_efficiency_designs.png",
        "E": "part3_v4_panel_E_top_balanced_designs.png",
        "F": "part3_v4_panel_F_efficiency_risk_smoothness_space.png",
        "G": "part3_v4_panel_G_best_nonsine_balanced_map.png",
        "H": "part3_v4_panel_H_pareto_efficiency_vs_risk.png",
        "I": "part3_v4_panel_I_monte_carlo_window_vs_balanced.png",
        "J": "part3_v4_panel_J_recommendation_matrix.png",
    },
    "P4": {
        "A": "part4_v4_panel_A_absolute_frequency_scaling.png",
        "B": "part4_v4_panel_B_period_scaling.png",
        "C": "part4_v4_panel_C_balanced_protocol_map.png",
        "D": "part4_v4_panel_D_best_balanced_class_by_tau.png",
        "E": "part4_v4_panel_E_top_balanced_protocol_candidates_tau20.png",
        "F": "part4_v4_panel_F_cycle_count_lookup_heatmap.png",
        "G": "part4_v4_panel_G_balanced_validation_matrix.png",
        "H": "part4_v4_panel_H_nonsine_balanced_alternative_ratio.png",
        "I": "part4_v4_panel_I_risk_smoothness_protocol_scatter.png",
        "J": "part4_v4_panel_J_frequency_lookup_table.png",
    },
    "P5": {
        "A": "part5_v3_panel_A_clinical_force_decay_profiles.png",
        "B": "part5_v3_panel_B_clinical_renewal_vs_part2_window.png",
        "C": "part5_v3_panel_C_required_microdynamic_period_by_tau.png",
        "D": "part5_v3_panel_D_force_availability_metrics.png",
        "E": "part5_v3_panel_E_part2_factor_clinical_vs_micro.png",
        "F": "part5_v3_panel_F_clinical_micro_period_scale_map.png",
        "G": "part5_v3_panel_G_clinical_renewal_event_schedules.png",
        "H": "part5_v3_panel_H_order_separation_tau_sensitivity.png",
        "I": "part5_v3_panel_I_manuscript_logic_bridge.png",
        "J": "part5_v3_panel_J_microdynamic_period_lookup_table.png",
    },
}


# ============================================================
# 4. Main figures
# ============================================================

MAIN_FIGURE_BLUEPRINTS = {
    "Figure_1_mechanical_signal_persistence": {
        "layout": (2, 2),
        "panels": [("P1", "C"), ("P1", "D"), ("P1", "F"), ("P1", "I")],
    },
    "Figure_2_frequency_window": {
        "layout": (2, 2),
        "panels": [("P2", "A"), ("P2", "B"), ("P2", "C"), ("P2", "G")],
    },
    "Figure_3_waveform_optimization": {
        "layout": (2, 2),
        "panels": [("P3", "B"), ("P3", "D"), ("P3", "E"), ("P3", "I")],
    },
    "Figure_4_protocol_translation": {
        "layout": (2, 2),
        "panels": [("P4", "A"), ("P4", "B"), ("P4", "D"), ("P4", "E")],
    },
    "Figure_5_clinical_timescale_bridge": {
        "layout": (2, 2),
        "panels": [("P5", "B"), ("P5", "E"), ("P5", "F"), ("P5", "H")],
    },
}


# ============================================================
# 5. Supplementary figures
# ============================================================

SUPPLEMENTARY_BLUEPRINTS = {
    "Figure_S1_part1_core_mechanical_foundation": {
        "layout": (3, 2),
        "panels": [("P1", x) for x in ["A", "B", "C", "D", "E", "F"]],
    },
    "Figure_S2_part1_secondary_maps": {
        "layout": (2, 2),
        "panels": [("P1", x) for x in ["G", "H", "I", "J"]],
    },
    "Figure_S3_part2_frequency_window_core": {
        "layout": (3, 2),
        "panels": [("P2", x) for x in ["A", "B", "C", "D", "E", "F"]],
    },
    "Figure_S4_part2_translation_sensitivity_summary": {
        "layout": (2, 2),
        "panels": [("P2", x) for x in ["G", "H", "I", "J"]],
    },
    "Figure_S5_part3_waveform_optimization_core": {
        "layout": (3, 2),
        "panels": [("P3", x) for x in ["A", "B", "C", "D", "E", "F"]],
    },
    "Figure_S6_part3_secondary_maps_and_recommendation": {
        "layout": (2, 2),
        "panels": [("P3", x) for x in ["G", "H", "I", "J"]],
    },
    "Figure_S7_part4_protocol_translation_core": {
        "layout": (3, 2),
        "panels": [("P4", x) for x in ["A", "B", "C", "D", "E", "F"]],
    },
    "Figure_S8_protocol_validation_and_clinical_bridge": {
        "layout": (4, 2),
        "panels": [
            ("P4", "G"), ("P4", "H"),
            ("P4", "I"), ("P4", "J"),
            ("P5", "B"), ("P5", "E"),
            ("P5", "F"), ("P5", "H"),
        ],
    },
}

QC_FULL_BLUEPRINTS = {
    "QC_P1_full_5x2_qc_overview": {
        "layout": (5, 2),
        "panels": [("P1", x) for x in "ABCDEFGHIJ"],
    },
    "QC_P2_full_5x2_qc_overview": {
        "layout": (5, 2),
        "panels": [("P2", x) for x in "ABCDEFGHIJ"],
    },
    "QC_P3_full_5x2_qc_overview": {
        "layout": (5, 2),
        "panels": [("P3", x) for x in "ABCDEFGHIJ"],
    },
    "QC_P4_full_5x2_qc_overview": {
        "layout": (5, 2),
        "panels": [("P4", x) for x in "ABCDEFGHIJ"],
    },
    "QC_P5_full_5x2_qc_overview": {
        "layout": (5, 2),
        "panels": [("P5", x) for x in "ABCDEFGHIJ"],
    },
}


# ============================================================
# 6. Figure legends
# ============================================================

FIGURE_LEGENDS = {
    "Figure 1": (
        "Viscoelastic mechanical foundation of time-structured loading. "
        "(A) Relaxation versus sustained activation under an idealized static reference and sinusoidal loading in the standard linear solid model. "
        "(B) Strain-rate signal persistence, showing decay of the static reference and repeated strain-rate peaks under dynamic loading. "
        "(C) Cumulative strain-rate signal across waveform classes under the same mechanical framework. "
        "(D) Raw efficacy-risk trade-off across viscoelastic models and waveform classes. "
        "This figure illustrates, within the adopted viscoelastic framework, that sustained force and sustained strain-rate stimulation are not equivalent model quantities."
    ),

    "Figure 2": (
        "Formation of a finite dimensionless loading window within the specified phenomenological model. "
        "(A) Frequency-response components using tau_ref = eta/E2. "
        "(B) Full-model candidate frequency window defined by the 90% score interval around the model-derived optimum. "
        "(C) Ablation analysis showing that the finite window emerges from the combined mechanical-drive, cellular-filtering, and overload-penalty terms. "
        "(D) Translation from dimensionless omega*tau to absolute frequency as a function of measured or assumed tau. "
        "The window should be interpreted as a model-derived, hypothesis-generating prediction rather than a measured biological constant."
    ),

    "Figure 3": (
        "Objective-dependent waveform ranking within the hard candidate dynamic window. "
        "(A) Hard-window constraint imported from the frequency-window model. "
        "(B) Top window-restricted efficiency designs. "
        "(C) Top balanced hard-window designs after incorporating mechanical-risk and smoothness surrogates. "
        "(D) Monte Carlo comparison between window-efficiency and balanced robustness. "
        "Within the specified objective functions, abrupt waveforms receive the highest window-restricted efficiency scores, whereas sinusoidal loading emerges as the conservative balanced candidate."
    ),

    "Figure 4": (
        "Translation of the dimensionless framework into experimental protocol space. "
        "(A) Dimensionless window translated into absolute frequency. "
        "(B) Loading period scaling with tau. "
        "(C) Best balanced experimental class across tau. "
        "(D) Top balanced protocol candidates at tau = 20 s. "
        "These outputs define tau-scaled experimental loading candidates and should not be interpreted as clinical prescriptions."
    ),

    "Figure 5": (
        "Clinical-scale renewal is separated from the model-derived micro-dynamic window by orders of magnitude. "
        "(A) Effective omega*tau values for representative clinical-scale renewal intervals compared with the candidate micro-dynamic window. "
        "(B) Imported full-model dynamic factor for representative clinical-scale renewal profiles and micro-dynamic candidates. "
        "(C) Period-scale map comparing clinical renewal periods with the model-derived micro-dynamic window. "
        "(D) Order-of-magnitude separation between clinical-scale renewal intervals and the lower bound of the candidate dynamic window across tau assumptions. "
        "The clinical profiles are representative modeling references, not universal clinical constants, and this comparison addresses time-scale separation rather than clinical efficacy."
    ),

    "Supplementary Figure S1": (
        "Extended mechanical-foundation outputs. "
        "Panels show input waveform archetypes, viscoelastic model comparison, static reference versus dynamic loading, strain-rate persistence, effective stimulation ratio, and cumulative strain-rate signal."
    ),

    "Supplementary Figure S2": (
        "Secondary mechanical-only maps and score matrices. "
        "Panels show activation-overload structure, signal decay, raw efficacy-risk distribution, and mechanical-only score matrix. "
        "These outputs are mechanical surrogates only and are not used as primary biological waveform rankings."
    ),

    "Supplementary Figure S3": (
        "Frequency-window construction and robustness. "
        "Panels show frequency-response components, candidate window, ablation analysis, cellular-cutoff sensitivity, mean-load design space, and Monte Carlo robustness. "
        "Together they show that, under the specified assumptions, the finite window is generated by the combined phenomenological framework and remains robust across parameter variation."
    ),

    "Supplementary Figure S4": (
        "Frequency-window translation and decision summary. "
        "Panels show absolute frequency scaling, phase/dissipation/high-frequency dynamics, cutoff sensitivity summary, and decision table. "
        "These analyses emphasize that absolute Hz requires a measured or assumed tau and should not be treated as a universal frequency."
    ),

    "Supplementary Figure S5": (
        "Core waveform-optimization design space. "
        "Panels show equal-RMS waveform candidates, hard candidate-window constraint, global versus hard-window efficiency, top efficiency designs, top balanced designs, and efficiency-risk-smoothness structure. "
        "The results demonstrate objective-dependent waveform ranking within the specified scoring framework."
    ),

    "Supplementary Figure S6": (
        "Secondary waveform maps, Pareto analysis, and recommendation matrix. "
        "Panels show best non-sine balanced alternatives, Pareto frontier, Monte Carlo robustness, and global/window/balanced recommendation matrix. "
        "These outputs support the interpretation that non-sine candidates are useful comparators and translational alternatives, not automatically superior clinical choices."
    ),

    "Supplementary Figure S7": (
        "Experimental protocol translation. "
        "Panels show frequency scaling, period scaling, balanced protocol map, best balanced class across tau, top candidates at tau = 20 s, and cycle-count lookup. "
        "The purpose is to convert dimensionless model predictions into experimentally testable loading protocols."
    ),

    "Supplementary Figure S8": (
        "Protocol validation and clinical time-scale bridge. "
        "Panels combine protocol-validation outputs with the most important clinical-scale comparison panels. "
        "They summarize the validation matrix, non-sine alternative ratio, risk-smoothness structure, frequency lookup, and the order-of-magnitude separation between representative clinical-scale renewal intervals and the model-derived micro-dynamic window."
    ),
}


# ============================================================
# 7. Source-level part-specific patches
# ============================================================

PART_SPECIFIC_PATCHES = {
    "P5": [
        (
            'ax.set_xlabel("Time (days)")',
            'ax.set_xlabel("Time (days)")\nax.set_xlim(0, 28)',
            1,
        ),
    ],
}


# ============================================================
# 8. Helper functions
# ============================================================

def find_source_script(part_id):
    for name in SOURCE_SCRIPT_ALIASES[part_id]:
        candidate = BASE_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find source script for {part_id}. Checked: {SOURCE_SCRIPT_ALIASES[part_id]}"
    )


# ============================================================
# 9. Source-code patching
# ============================================================

def build_publication_style_injection():
    return f'''
# ============================================================
# Injected by Part 6 v13: publication-grade style overrides
# ============================================================

import textwrap
from matplotlib.axes import Axes as _Part6Axes
from matplotlib.figure import Figure as _Part6Figure

_PART6_TITLE_SIZE = {STYLE["title_font_size"]}
_PART6_AXIS_SIZE = {STYLE["axis_font_size"]}
_PART6_TICK_SIZE = {STYLE["tick_font_size"]}
_PART6_LEGEND_SIZE = {STYLE["legend_font_size"]}
_PART6_TABLE_SIZE = {STYLE["table_font_size"]}

try:
    from cycler import cycler
    plt.rcParams["axes.prop_cycle"] = cycler(color={FORCED_PROP_CYCLE})
except Exception:
    pass

plt.rcParams.update({{
    "axes.titlesize": _PART6_TITLE_SIZE,
    "axes.labelsize": _PART6_AXIS_SIZE,
    "xtick.labelsize": _PART6_TICK_SIZE,
    "ytick.labelsize": _PART6_TICK_SIZE,
    "legend.fontsize": _PART6_LEGEND_SIZE,
    "lines.linewidth": {STYLE["line_width_default"]},
    "axes.linewidth": {STYLE["axis_linewidth"]},
    "xtick.major.width": {STYLE["major_tick_width"]},
    "ytick.major.width": {STYLE["major_tick_width"]},
}})


# ------------------------------------------------------------
# Force panel titles to match axis-label size.
# This overrides explicit small fontsize values in Part 1-5 source scripts.
# ------------------------------------------------------------

_PART6_ORIG_SET_TITLE = _Part6Axes.set_title

def _part6_publication_set_title(self, label, *args, **kwargs):
    kwargs["fontsize"] = max(
        float(kwargs.get("fontsize", _PART6_AXIS_SIZE)),
        _PART6_AXIS_SIZE
    )
    kwargs["fontweight"] = kwargs.get("fontweight", "normal")
    return _PART6_ORIG_SET_TITLE(self, label, *args, **kwargs)

_Part6Axes.set_title = _part6_publication_set_title




# ------------------------------------------------------------
# Suppress original source panel labels.
# This catches both fig.text(...) and ax.text(...).
# Composite panel labels are added only once during figure assembly.
# ------------------------------------------------------------

def _part6_is_panel_letter(s):
    """Robustly identify source panel letters, including math/bold variants."""
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t:
        return False
    # Normalize common matplotlib/mathtext/bold wrappers used for panel tags.
    for token in ["$", "\\bf", "\\mathbf", "\\mathrm", "\\textbf", "{", "}", " ", "\\n", "\\t"]:
        t = t.replace(token, "")
    t = t.strip(".:：)")
    return t in list("ABCDEFGHIJ")

_PART6_ORIG_FIGURE_TEXT = _Part6Figure.text

def _part6_publication_figure_text(self, x, y, s, *args, **kwargs):
    if _part6_is_panel_letter(s):
        try:
            sx = float(x)
            sy = float(y)
            if -0.20 <= sx <= 0.45 and 0.35 <= sy <= 1.20:
                return _PART6_ORIG_FIGURE_TEXT(self, x, y, "", *args, **kwargs)
        except Exception:
            return _PART6_ORIG_FIGURE_TEXT(self, x, y, "", *args, **kwargs)

    return _PART6_ORIG_FIGURE_TEXT(self, x, y, s, *args, **kwargs)

_Part6Figure.text = _part6_publication_figure_text


_PART6_ORIG_AXES_TEXT = _Part6Axes.text

def _part6_publication_axes_text(self, x, y, s, *args, **kwargs):
    if _part6_is_panel_letter(s):
        try:
            sx = float(x)
            sy = float(y)

            # Usual panel-letter locations in axes coordinates are around
            # x = -0.15 to 0.05 and y = 0.9 to 1.1.
            if -0.80 <= sx <= 0.55 and 0.35 <= sy <= 1.70:
                return _PART6_ORIG_AXES_TEXT(self, x, y, "", *args, **kwargs)
        except Exception:
            return _PART6_ORIG_AXES_TEXT(self, x, y, "", *args, **kwargs)

    return _PART6_ORIG_AXES_TEXT(self, x, y, s, *args, **kwargs)

_Part6Axes.text = _part6_publication_axes_text


# ------------------------------------------------------------
# Legend relocation.
# ------------------------------------------------------------

_PART6_ORIG_LEGEND = _Part6Axes.legend

def _part6_publication_legend(self, *args, **kwargs):
    kwargs["fontsize"] = max(float(kwargs.get("fontsize", _PART6_LEGEND_SIZE)), _PART6_LEGEND_SIZE)
    kwargs["frameon"] = False
    kwargs["borderaxespad"] = kwargs.get("borderaxespad", 0.0)
    kwargs["handlelength"] = kwargs.get("handlelength", 2.0)
    kwargs["handletextpad"] = kwargs.get("handletextpad", 0.6)

    loc0 = kwargs.get("loc", None)

    if loc0 in [None, "best", "upper right", "lower right", "upper left", "lower left", "center right"]:
        kwargs["loc"] = "center left"
        kwargs["bbox_to_anchor"] = (1.02, 0.5)

    return _PART6_ORIG_LEGEND(self, *args, **kwargs)

_Part6Axes.legend = _part6_publication_legend


# ------------------------------------------------------------
# Tick-label wrapping.
# ------------------------------------------------------------

_PART6_ORIG_SET_XTICKLABELS = _Part6Axes.set_xticklabels

def _part6_set_xticklabels(self, labels, *args, **kwargs):
    new_labels = []

    for lab in labels:
        s = lab.get_text() if hasattr(lab, "get_text") else str(lab)

        s = s.replace("Micro-dynamic ", "Micro-" + chr(10) + "dynamic ")
        s = s.replace("micro-dynamic ", "micro-" + chr(10) + "dynamic ")
        s = s.replace("Removable appliance ", "Removable" + chr(10) + "appliance ")
        s = s.replace("Fixed appliance ", "Fixed" + chr(10) + "appliance ")
        s = s.replace("Fast renewal ", "Fast" + chr(10) + "renewal ")
        s = s.replace("1-day renewal", "Daily" + chr(10) + "renewal")
        s = s.replace("Window efficiency best", "Window" + chr(10) + "efficiency best")
        s = s.replace("Balanced best", "Balanced" + chr(10) + "best")

        s = s.replace("omega=", chr(10) + "ωτ=")
        s = s.replace("wτ=", chr(10) + "ωτ=")
        s = s.replace("wr=", chr(10) + "ωτ=")

        if len(s) > 18 and chr(10) not in s:
            s = textwrap.fill(s, width=14)

        new_labels.append(s)

    kwargs["fontsize"] = max(float(kwargs.get("fontsize", _PART6_TICK_SIZE)), _PART6_TICK_SIZE)

    if "rotation" not in kwargs:
        kwargs["rotation"] = {STYLE["default_x_tick_rotation"]}
    if "ha" not in kwargs:
        kwargs["ha"] = "right"

    return _PART6_ORIG_SET_XTICKLABELS(self, new_labels, *args, **kwargs)

_Part6Axes.set_xticklabels = _part6_set_xticklabels


_PART6_ORIG_SET_YTICKLABELS = _Part6Axes.set_yticklabels

def _part6_set_yticklabels(self, labels, *args, **kwargs):
    new_labels = []

    for lab in labels:
        s = lab.get_text() if hasattr(lab, "get_text") else str(lab)

        s = s.replace("Balanced dynamic candidate", "Balanced candidate")
        s = s.replace("Micro-dynamic Part 3 balanced", "Balanced candidate")
        s = s.replace("Dynamic-window center", "Dynamic window center")
        s = s.replace("Micro-dynamic Part 2 center", "Dynamic window center")
        s = s.replace("Micro-dynamic ", "Micro-" + chr(10) + "dynamic ")
        s = s.replace("micro-dynamic ", "micro-" + chr(10) + "dynamic ")
        s = s.replace("Removable appliance ", "Removable" + chr(10) + "appliance ")
        s = s.replace("Fixed appliance ", "Fixed" + chr(10) + "appliance ")
        s = s.replace("Fast renewal ", "Fast" + chr(10) + "renewal ")
        s = s.replace("1-day renewal", "Daily" + chr(10) + "renewal")

        if len(s) > 24 and chr(10) not in s:
            s = textwrap.fill(s, width=18)

        new_labels.append(s)

    kwargs["fontsize"] = max(float(kwargs.get("fontsize", _PART6_TICK_SIZE)), _PART6_TICK_SIZE)

    return _PART6_ORIG_SET_YTICKLABELS(self, new_labels, *args, **kwargs)

_Part6Axes.set_yticklabels = _part6_set_yticklabels



# ------------------------------------------------------------
# Annotate suppression for source panel letters.
# ------------------------------------------------------------

_PART6_ORIG_AXES_ANNOTATE = _Part6Axes.annotate

def _part6_publication_axes_annotate(self, text, xy, *args, **kwargs):
    if _part6_is_panel_letter(text):
        try:
            sx = float(xy[0])
            sy = float(xy[1])
            if -0.80 <= sx <= 0.55 and 0.35 <= sy <= 1.70:
                return _PART6_ORIG_AXES_ANNOTATE(self, "", xy, *args, **kwargs)
        except Exception:
            return _PART6_ORIG_AXES_ANNOTATE(self, "", xy, *args, **kwargs)

    return _PART6_ORIG_AXES_ANNOTATE(self, text, xy, *args, **kwargs)

_Part6Axes.annotate = _part6_publication_axes_annotate

# ------------------------------------------------------------
# Savefig override.
# ------------------------------------------------------------

_PART6_ORIG_SAVEFIG = _Part6Figure.savefig

def _part6_savefig(self, fname, *args, **kwargs):
    kwargs["dpi"] = {STYLE["dpi"]}
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.36)
    kwargs.setdefault("facecolor", BG if "BG" in globals() else "#FAFAFA")
    return _PART6_ORIG_SAVEFIG(self, fname, *args, **kwargs)

_Part6Figure.savefig = _part6_savefig

'''


def suppress_source_panel_labels_by_regex(text):
    """
    v13 source-level panel-letter suppression.

    This is the clean solution for the manuscript Part 1-5 scripts:
    each source file defines the same helper:

        def panel_label(ax, label):
            ax.text(...)

    We do NOT erase rendered PNG pixels. We only patch the temporary source copy
    inside Part 6 so panel_label becomes a no-op. The manuscript original Part 1-5
    files and their normal ABCDEFG outputs are not modified.
    """

    # Exact common helper used by Part 1-5. Stop at the next non-indented line.
    text, n = re.subn(
        r'(?m)^def\s+panel_label\s*\(\s*ax\s*,\s*label\s*\)\s*:\n(?:[ \t]+.*\n)+',
        'def panel_label(ax, label):\n    return None  # Part 6 v13: source panel label suppressed in patched copy only\n\n',
        text,
        count=1,
    )

    # Conservative fallback for helper-name variants, without replacing calls.
    # Calls can remain because the helper is now a no-op.
    for name in ["add_panel_label", "label_panel", "add_panel_letter", "panel_letter", "annotate_panel"]:
        text = re.sub(
            rf'(?m)^def\s+{name}\s*\([^)]*\)\s*:\n(?:[ \t]+.*\n)+',
            f'def {name}(*args, **kwargs):\n    return None  # Part 6 v13: source panel-label helper suppressed\n\n',
            text,
            count=1,
        )

    return text

def rewrite_public_scientific_labels(text):
    replacements = {
        # ====================================================
        # Broad public-facing replacements for internal Part labels
        # ====================================================

        # Part 1
        "Part 1 mechanical foundation": "mechanical-foundation model",
        "Part 1 mechanical-only": "mechanical-only model",
        "Part 1": "mechanical-foundation model",

        # Part 2 / Part2 / P2
        "Hard Part 2 window constraint": "Hard candidate dynamic-window constraint",
        "Hard Part2 window constraint": "Hard candidate dynamic-window constraint",
        "Part 2 90% window": "Candidate dynamic window",
        "Part2 90% window": "Candidate dynamic window",
        "Part 2 micro-dynamic window": "Candidate dynamic window",
        "Part2 micro-dynamic window": "Candidate dynamic window",
        "Part 2 candidate window": "Candidate dynamic window",
        "Part2 candidate window": "Candidate dynamic window",
        "Part 2 v3 90% window": "Candidate dynamic window",
        "Part2 v3 90% window": "Candidate dynamic window",
        "Part 2 v3 window": "Candidate dynamic window",
        "Part2 v3 window": "Candidate dynamic window",
        "Part 2 window": "candidate dynamic window",
        "Part2 window": "candidate dynamic window",
        "Part 2 frequency window": "candidate dynamic window",
        "Part2 frequency window": "candidate dynamic window",
        "Part 2 model window": "frequency-window model window",
        "Part2 model window": "frequency-window model window",
        "Part 2 full-model factor": "Dynamic-window factor",
        "Part2 full-model factor": "Dynamic-window factor",
        "Part 2 v3 full-model factor": "Dynamic-window factor",
        "Part2 v3 full-model factor": "Dynamic-window factor",
        "Imported Part 2 v3 full-model factor": "Imported dynamic-window factor",
        "Imported Part2 v3 full-model factor": "Imported dynamic-window factor",
        "Part 2 factor distinguishes clinical-scale renewal from micro stimulation":
            "Model score separates clinical-scale renewal from dynamic loading",
        "Part2 factor distinguishes clinical-scale renewal from micro stimulation":
            "Model score separates clinical-scale renewal from dynamic loading",
        "Part 2 frequency-window model": "frequency-window model",
        "Part2 frequency-window model": "frequency-window model",
        "Part 2 framework": "frequency-window framework",
        "Part2 framework": "frequency-window framework",

        # Part 3 / Part3 / P3
        "Part 3 balanced": "Balanced candidate",
        "Part3 balanced": "Balanced candidate",
        "Part 3 waveform optimization": "waveform-optimization model",
        "Part3 waveform optimization": "waveform-optimization model",
        "Part 3 v4 balanced": "Balanced candidate",
        "Part3 v4 balanced": "Balanced candidate",
        "Part 3 v4": "waveform-optimization model",
        "Part3 v4": "waveform-optimization model",
        "Part 3": "waveform-optimization model",
        "Part3": "waveform-optimization model",

        # Part 4
        "Part 4 protocol translation": "protocol-translation model",
        "Part 4 v4": "protocol-translation model",
        "Part 4": "protocol-translation model",

        # Part 5
        "Part 5 clinical bridge": "clinical-timescale bridge",
        "Part 5": "clinical-timescale bridge",

        # ====================================================
        # Specific tick-label / legend / table cleanup
        # ====================================================

        "Dynamic-window center": "Dynamic window center",
        "Micro-dynamic Part 2 center": "Dynamic window center",
        "Balanced dynamic candidate": "Balanced dynamic candidate",
        "Micro-dynamic Part 3 balanced": "Balanced dynamic candidate",

        "P2 center period": "Dynamic-window center period",
        "P3 balanced period": "Balanced candidate period",
        "P2 center": "Dynamic-window center",
        "P3 balanced": "Balanced candidate",

        "Part2 center": "Dynamic-window center",
        "Part 2 center": "Dynamic-window center",
        "Part3 balanced": "Balanced candidate",
        "Part 3 balanced": "Balanced candidate",

        # Common generated figure-title phrases
        "Clinical renewal remains orders below the Part 2 window":
            "Clinical renewal remains orders below the candidate dynamic window",
        "Clinical renewal remains orders below the Part2 window":
            "Clinical renewal remains orders below the candidate dynamic window",
        "Clinical renewal intervals are far below the Part 2 window":
            "Clinical renewal intervals are far below the candidate dynamic window",
        "Clinical renewal intervals are far below the Part2 window":
            "Clinical renewal intervals are far below the candidate dynamic window",
        "Clinical renewal and micro-dynamic loading occupy different Part 2 time scales":
            "Clinical renewal and micro-dynamic loading occupy distinct time scales",
    }

    for old in sorted(replacements.keys(), key=len, reverse=True):
        text = text.replace(old, replacements[old])

    return text


def replace_outdirs_and_dependencies(text, part_id):
    """
    v13: run temporary patched copies of Part 1-5 into Part 6 controlled folders.

    Important:
    - The manuscript original Part 1-5 scripts are not edited.
    - Their normal original output folders can remain with ABCDEFG labels.
    - Only the patched copies created under Part 6 write no-letter panels into
      figure_assembly_outputs/source_regenerated_outputs.
    - Cross-part dependencies are mapped correctly: Part 3 reads patched Part 2,
      Part 4 reads patched Part 2/3, and Part 5 reads patched Part 2/3.
    """
    new_folder = PATCHED_OUTPUT_FOLDERS[part_id]

    # Correct dependency remapping: each original output folder is replaced by
    # that same part's patched output folder, not by the current part's folder.
    for old_part, old_folder in ORIGINAL_OUTPUT_FOLDERS.items():
        text = text.replace(old_folder, PATCHED_OUTPUT_FOLDERS[old_part])

    # Force the current script's primary OUTDIR-like assignment.
    output_var_pattern = re.compile(
        r'(?m)^(\s*(?:OUTDIR|OUT_DIR|OUTPUT_DIR|OUTPUT_FOLDER|SAVE_DIR|FIGURE_DIR|FIG_DIR)\s*=\s*)'
        r'(?:Path\s*\(\s*)?[rRuUbBfF]*["\'][^"\']+["\']\s*\)?\s*$'
    )

    def _replace_assignment(m):
        return f'{m.group(1)}r"{new_folder}"'

    text, _ = output_var_pattern.subn(_replace_assignment, text, count=1)

    return text

def replace_common_style(text):
    text = re.sub(r'DPI\s*=\s*300', f'DPI = {STYLE["dpi"]}', text)
    text = re.sub(r'COMBINED_DPI\s*=\s*600', f'COMBINED_DPI = {STYLE["dpi"]}', text)

    text = re.sub(
        r'FIGSIZE\s*=\s*\([^)]+\)',
        f'FIGSIZE = ({STYLE["fig_width_in"]}, {STYLE["fig_height_in"]})',
        text,
    )

    text = re.sub(r'"font\.size":\s*8', f'"font.size": {STYLE["base_font_size"]}', text)
    text = re.sub(r'"axes\.linewidth":\s*0\.8', f'"axes.linewidth": {STYLE["axis_linewidth"]}', text)
    text = re.sub(r'"xtick\.major\.width":\s*0\.7', f'"xtick.major.width": {STYLE["major_tick_width"]}', text)
    text = re.sub(r'"ytick\.major\.width":\s*0\.7', f'"ytick.major.width": {STYLE["major_tick_width"]}', text)

    text = re.sub(
        r'ax\.tick_params\(labelsize=7\)',
        f'ax.tick_params(labelsize={STYLE["tick_font_size"]})',
        text,
    )

    text = text.replace('fig.tight_layout(pad=0.8)', 'fig.tight_layout(pad=1.7)')
    text = text.replace('fig.tight_layout(pad=0.6)', 'fig.tight_layout(pad=1.7)')

    def fontsize_repl(match):
        val = float(match.group(1))

        if val < 6:
            new_val = 9.5
        elif val < 7:
            new_val = 10.5
        elif val < 8:
            new_val = 11.5
        elif val < 9:
            new_val = 12.5
        elif val < 11:
            new_val = 13.5
        elif val < 13:
            new_val = 14.5
        else:
            new_val = val

        return f"fontsize={new_val:g}"

    text = re.sub(r'fontsize=([0-9]+(?:\.[0-9]+)?)', fontsize_repl, text)

    text = re.sub(
        r'table\.set_fontsize\([0-9]+(?:\.[0-9]+)?\)',
        f'table.set_fontsize({STYLE["table_font_size"]})',
        text,
    )

    text = re.sub(
        r'legend\(fontsize=([0-9]+(?:\.[0-9]+)?)',
        f'legend(fontsize={STYLE["legend_font_size"]}',
        text,
    )

    text = text.replace('ncol=2', 'ncol=1')

    for old, new in COLOR_REWRITE.items():
        text = text.replace(old, new)

    # v8: remove old source panel labels before runtime style injection.
    text = suppress_source_panel_labels_by_regex(text)

    # v8: rewrite public-facing labels more completely.
    text = rewrite_public_scientific_labels(text)

    injection = build_publication_style_injection()

    if "Injected by Part 6 v13" not in text:
        pattern = r'(plt\.rcParams\.update\(\{.*?\}\))'

        # IMPORTANT: use a replacement function, not a replacement string.
        # The injected code contains regex backslashes such as \m, \s, \n.
        # In Python 3.13, passing those through re.subn's replacement-string
        # parser raises PatternError: bad escape \m.
        text2, n = re.subn(
            pattern,
            lambda m: m.group(1) + "\n" + injection,
            text,
            count=1,
            flags=re.DOTALL,
        )

        if n > 0:
            text = text2
        else:
            text2, n = re.subn(
                r'(import\s+matplotlib\.pyplot\s+as\s+plt\s*)',
                lambda m: m.group(1) + "\n" + injection,
                text,
                count=1,
            )

            if n > 0:
                text = text2
            else:
                text = injection + "\n" + text

    return text


def apply_part_specific_patches(text, part_id):
    for patch in PART_SPECIFIC_PATCHES.get(part_id, []):
        if len(patch) == 3:
            old, new, count = patch
            text = text.replace(old, new, count)
        else:
            old, new = patch
            text = text.replace(old, new)

    return text


def patch_source_script(part_id):
    source_path = find_source_script(part_id)
    text = source_path.read_text(encoding="utf-8", errors="ignore")

    text = replace_outdirs_and_dependencies(text, part_id)
    text = replace_common_style(text)
    text = apply_part_specific_patches(text, part_id)

    patched_path = PATCHED_SCRIPT_DIR / f"{part_id}_publication_grade_v13_patched.py"
    patched_path.write_text(text, encoding="utf-8")

    return source_path, patched_path


# ============================================================
# 10. Image refinement
# ============================================================

def trim_image(img, bg_rgb=(250, 250, 250), tolerance=10):
    img = img.convert("RGB")
    bg = Image.new("RGB", img.size, bg_rgb)
    diff = ImageChops.difference(img, bg)
    diff = ImageOps.grayscale(diff)
    bbox = diff.point(lambda p: 255 if p > tolerance else 0).getbbox()

    if bbox is None:
        return img

    left, top, right, bottom = bbox

    safety = 16
    left = max(0, left - safety)
    top = max(0, top - safety)
    right = min(img.size[0], right + safety)
    bottom = min(img.size[1], bottom + safety)

    return img.crop((left, top, right, bottom))


def pad_to_aspect(img, target_w, target_h, bg_rgb):
    img = img.convert("RGB")
    target_aspect = target_w / target_h

    w, h = img.size
    current = w / h

    if current < target_aspect:
        new_w = int(math.ceil(h * target_aspect))
        pad = new_w - w
        img = ImageOps.expand(img, border=(pad // 2, 0, pad - pad // 2, 0), fill=bg_rgb)
    elif current > target_aspect:
        new_h = int(math.ceil(w / target_aspect))
        pad = new_h - h
        img = ImageOps.expand(img, border=(0, pad // 2, 0, pad - pad // 2), fill=bg_rgb)

    return img


def resize_exact(img, target_w, target_h):
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS

    return img.resize((target_w, target_h), resample_filter)

def erase_possible_source_panel_label(img, bg_rgb=(250, 250, 250)):
    """
    v13: disabled.

    v10's component-based raster eraser damaged real title/axis text.
    We no longer modify rendered panel pixels. Old source labels must be removed
    before rendering via source-code/runtime suppression.
    """
    return img.convert("RGB")

def refined_panel_name(part_id, panel_letter):
    src = PANEL_FILES[part_id][panel_letter]
    return f"{part_id}_{panel_letter}_{Path(src).stem}_600dpi_5x3.png"


def refined_panel_path(part_id, panel_letter):
    return REFINED_PANEL_DIR / part_id / refined_panel_name(part_id, panel_letter)


def find_source_panel_file(part_id, panel_letter):
    """
    v9 robust source-panel resolver.

    Primary path: controlled patched-output folder.
    Fallback: exact filename search under BASE_DIR, newest file first.

    This prevents blank composite panels when a source script silently writes to
    its original output folder or a differently named output variable.
    """
    expected = Path(PATCHED_OUTPUT_FOLDERS[part_id]) / PANEL_FILES[part_id][panel_letter]
    if expected.exists():
        return expected, "OK"

    filename = PANEL_FILES[part_id][panel_letter]
    candidates = []

    search_roots = [OUTDIR, BASE_DIR]
    seen = set()

    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        try:
            for cand in root.rglob(filename):
                try:
                    resolved = cand.resolve()
                except Exception:
                    resolved = cand
                if resolved in seen:
                    continue
                seen.add(resolved)

                # Avoid using Part 6's primary composite figures as source panels.
                s = str(cand)
                if "main_figures" in s or "supplementary_figures" in s or "qc_full" in s:
                    continue
                if "refined_single_panels" in s:
                    continue

                try:
                    mtime = cand.stat().st_mtime
                except Exception:
                    mtime = 0
                candidates.append((mtime, cand))
        except Exception:
            pass

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        newest = candidates[0][1]
        return newest, "OK_FALLBACK_NEWEST"

    return expected, "MISSING"


def regenerate_single_panel_from_source_output(part_id, panel_letter):
    src_path, source_status = find_source_panel_file(part_id, panel_letter)

    out_folder = REFINED_PANEL_DIR / part_id
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = refined_panel_path(part_id, panel_letter)

    if not src_path.exists():
        return {
            "Part": part_id,
            "Panel": panel_letter,
            "Expected_source_panel": str(Path(PATCHED_OUTPUT_FOLDERS[part_id]) / PANEL_FILES[part_id][panel_letter]),
            "Resolved_source_panel": str(src_path),
            "Output_panel": str(out_path),
            "Status": "MISSING",
        }

    img = Image.open(src_path).convert("RGB")

    # v10: first remove source labels in native regenerated panel space.
    img = erase_possible_source_panel_label(img, bg_rgb=STYLE["bg_rgb"])

    if STYLE["trim_output"]:
        img = trim_image(img, bg_rgb=STYLE["bg_rgb"], tolerance=STYLE["trim_tolerance"])

    img = ImageOps.expand(img, border=STYLE["safe_margin_px"], fill=STYLE["bg_rgb"])
    img = pad_to_aspect(img, STYLE["single_panel_width_px"], STYLE["single_panel_height_px"], STYLE["bg_rgb"])
    img = resize_exact(img, STYLE["single_panel_width_px"], STYLE["single_panel_height_px"])

    # v9 fail-safe: remove any remaining old source A-J letter before composite assembly.
    img = erase_possible_source_panel_label(img, bg_rgb=STYLE["bg_rgb"])

    img.save(out_path, dpi=(STYLE["single_panel_dpi"], STYLE["single_panel_dpi"]))

    return {
        "Part": part_id,
        "Panel": panel_letter,
        "Expected_source_panel": str(Path(PATCHED_OUTPUT_FOLDERS[part_id]) / PANEL_FILES[part_id][panel_letter]),
        "Resolved_source_panel": str(src_path),
        "Output_panel": str(out_path),
        "Status": source_status,
    }

# ============================================================
# 11. Composite assembly
# ============================================================

def load_panel(part_id, panel_letter):
    p = refined_panel_path(part_id, panel_letter)

    if p.exists():
        return Image.open(p).convert("RGB")

    img = Image.new(
        "RGB",
        (STYLE["single_panel_width_px"], STYLE["single_panel_height_px"]),
        STYLE["bg_rgb"],
    )

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", 88)
    except Exception:
        font = ImageFont.load_default()

    msg = f"Missing {part_id}{panel_letter}"
    bbox = draw.textbbox((0, 0), msg, font=font)

    draw.text(
        ((img.size[0] - (bbox[2] - bbox[0])) // 2, img.size[1] // 2),
        msg,
        fill=(70, 70, 70),
        font=font,
    )

    return img


def get_label_font(size):
    try:
        import matplotlib.font_manager as fm
        prop = fm.FontProperties(family="DejaVu Sans", weight="bold")
        font_path = fm.findfont(prop, fallback_to_default=True)
        return ImageFont.truetype(font_path, size)
    except Exception:
        pass

    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "Arial Bold.ttf",
        "Arial.ttf",
    ]

    for f in font_candidates:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass

    return ImageFont.load_default()


def draw_composite_panel_labels(canvas, label_positions, letters):
    draw = ImageDraw.Draw(canvas)
    font = get_label_font(STYLE["composite_panel_label_size"])

    for (x0, y0), letter in zip(label_positions, letters):
        x = x0 + STYLE["composite_panel_label_x"]
        y = y0 + STYLE["composite_panel_label_y"]

        draw.text(
            (x, y),
            letter,
            fill=(35, 35, 35),
            font=font,
        )


def compose_grid(panel_refs, layout, output_path, mode="main", relabel=True):
    rows, cols = layout

    panel_w = STYLE["single_panel_width_px"]
    panel_h = STYLE["single_panel_height_px"]

    if mode == "main":
        gap = STYLE["main_gap_px"]
        outer = STYLE["main_outer_pad_px"]
        border = STYLE["main_border_px"]
        dpi = STYLE["main_composite_dpi"]
    else:
        gap = STYLE["supp_gap_px"]
        outer = STYLE["supp_outer_pad_px"]
        border = STYLE["supp_border_px"]
        dpi = STYLE["supp_composite_dpi"]

    total_w = outer * 2 + cols * panel_w + (cols - 1) * gap
    total_h = outer * 2 + rows * panel_h + (rows - 1) * gap

    canvas = Image.new("RGB", (total_w, total_h), STYLE["paper_rgb"])
    label_positions = []

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols

        x = outer + c * (panel_w + gap)
        y = outer + r * (panel_h + gap)

        if idx < len(panel_refs):
            part_id, panel_letter = panel_refs[idx]
            panel = load_panel(part_id, panel_letter)
        else:
            panel = Image.new("RGB", (panel_w, panel_h), STYLE["bg_rgb"])

        if border > 0:
            panel = ImageOps.expand(panel, border=border, fill=STYLE["border_rgb"])
            panel = resize_exact(panel, panel_w, panel_h)

        canvas.paste(panel, (x, y))

        if idx < len(panel_refs):
            label_positions.append((x, y))

    if relabel:
        letters = [chr(ord("A") + i) for i in range(len(panel_refs))]
        draw_composite_panel_labels(canvas, label_positions, letters)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(dpi, dpi))

    return {
        "Output": str(output_path),
        "Rows": rows,
        "Cols": cols,
        "Panels": ", ".join([f"{p}{l}" for p, l in panel_refs]),
        "Composite_labels": ", ".join([chr(ord("A") + i) for i in range(len(panel_refs))]),
        "Status": "OK",
    }


# ============================================================
# 12. Legends
# ============================================================

def write_figure_legends():
    legend_txt = LEGEND_DIR / "part6_v9_figure_legends.txt"
    legend_md = LEGEND_DIR / "part6_v9_figure_legends.md"

    lines = []
    lines.append("Part 6 v13 Figure Legends")
    lines.append("=" * 80)
    lines.append("")

    for key, value in FIGURE_LEGENDS.items():
        lines.append(key)
        lines.append("-" * len(key))
        lines.append(value)
        lines.append("")

    legend_txt.write_text("\n".join(lines), encoding="utf-8")

    md_lines = []
    md_lines.append("# Part 6 v13 Figure Legends")
    md_lines.append("")

    for key, value in FIGURE_LEGENDS.items():
        md_lines.append(f"## {key}")
        md_lines.append("")
        md_lines.append(value)
        md_lines.append("")

    legend_md.write_text("\n".join(md_lines), encoding="utf-8")

    return legend_txt, legend_md


# ============================================================
# 13. Run patched scripts
# ============================================================

def run_patched_scripts():
    patched_records = []

    for part_id in ["P1", "P2", "P3", "P4", "P5"]:
        source_path, patched_path = patch_source_script(part_id)

        print(f"\nRunning patched {part_id}: {patched_path}")
        subprocess.run(["python", str(patched_path)], cwd=str(BASE_DIR), check=True)

        patched_records.append({
            "Part": part_id,
            "Original_script": str(source_path),
            "Patched_script": str(patched_path),
            "Patched_output_folder": PATCHED_OUTPUT_FOLDERS[part_id],
            "Status": "RAN_OK",
        })

    df = pd.DataFrame(patched_records)
    df.to_csv(MANIFEST_DIR / "part6_v9_patched_script_manifest.csv", index=False)

    return df


# ============================================================
# 14. Style manifest
# ============================================================

def write_style_manifest():
    rows = []

    for k, v in STYLE.items():
        rows.append({
            "Section": "STYLE",
            "Key": k,
            "Value": str(v),
        })

    for old, new in COLOR_REWRITE.items():
        rows.append({
            "Section": "COLOR_REWRITE",
            "Key": old,
            "Value": new,
        })

    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST_DIR / "part6_v9_style_manifest.csv", index=False)

    return df


# ============================================================
# 15. Main workflow
# ============================================================

def main():
    if not PIL_AVAILABLE:
        raise ImportError("Pillow is required. Install it with: pip install pillow")

    print("\n" + "=" * 90)
    print("Part 6 FIGURE ASSEMBLY v13")
    print("Main figures: Figure 1-5; Supplementary figures: S1-S8")
    print("Original source labels are suppressed; composite labels are regenerated cleanly.")
    print("=" * 90)

    print("\nStep 1: patch and rerun Part 1-5 source scripts.")
    run_patched_scripts()

    print("\nStep 2: collect and refine every single panel.")
    panel_records = []

    for part_id in ["P1", "P2", "P3", "P4", "P5"]:
        for panel_letter in "ABCDEFGHIJ":
            rec = regenerate_single_panel_from_source_output(part_id, panel_letter)
            panel_records.append(rec)

    df_panels = pd.DataFrame(panel_records)
    df_panels.to_csv(MANIFEST_DIR / "part6_v9_refined_single_panel_manifest.csv", index=False)

    print("\nStep 3: build main manuscript figures, Figure 1-5.")
    main_records = []

    for figure_name, blueprint in MAIN_FIGURE_BLUEPRINTS.items():
        out_path = MAIN_FIGURE_DIR / f"{figure_name}_600dpi.png"

        rec = compose_grid(
            panel_refs=blueprint["panels"],
            layout=blueprint["layout"],
            output_path=out_path,
            mode="main",
            relabel=True,
        )

        rec["Figure"] = figure_name
        rec["Type"] = "Main"
        main_records.append(rec)

    df_main = pd.DataFrame(main_records)
    df_main.to_csv(MANIFEST_DIR / "part6_v9_main_figure_manifest.csv", index=False)

    print("\nStep 4: build submission supplementary figures, S1-S8.")
    supp_records = []

    for figure_name, blueprint in SUPPLEMENTARY_BLUEPRINTS.items():
        out_path = SUPPLEMENTARY_FIGURE_DIR / f"{figure_name}_600dpi.png"

        rec = compose_grid(
            panel_refs=blueprint["panels"],
            layout=blueprint["layout"],
            output_path=out_path,
            mode="supp",
            relabel=True,
        )

        rec["Figure"] = figure_name
        rec["Type"] = "Supplementary submission"
        supp_records.append(rec)

    df_supp = pd.DataFrame(supp_records)
    df_supp.to_csv(MANIFEST_DIR / "part6_v9_supplementary_S1_to_S8_manifest.csv", index=False)

    print("\nStep 5: build optional QC full-overview figures for quality control.")
    qc_records = []

    for figure_name, blueprint in QC_FULL_BLUEPRINTS.items():
        out_path = QC_FULL_SUPPLEMENTARY_DIR / f"{figure_name}_600dpi.png"

        rec = compose_grid(
            panel_refs=blueprint["panels"],
            layout=blueprint["layout"],
            output_path=out_path,
            mode="supp",
            relabel=True,
        )

        rec["Figure"] = figure_name
        rec["Type"] = "Quality-control overview"
        qc_records.append(rec)

    df_qc = pd.DataFrame(qc_records)
    df_qc.to_csv(MANIFEST_DIR / "part6_v9_qc_full_overview_manifest.csv", index=False)

    print("\nStep 6: write figure legends.")
    legend_txt, legend_md = write_figure_legends()

    print("\nStep 7: write style manifest.")
    write_style_manifest()

    print("\n" + "=" * 90)
    print("Part 6 v13 completed.")
    print("=" * 90)

    print("\nOutput folder:")
    print(OUTDIR)

    print("\nRefined single panels:")
    print(REFINED_PANEL_DIR)

    print("\nMain figures, Figure 1-5:")
    for p in sorted(MAIN_FIGURE_DIR.glob("*.png")):
        print(p.name)

    print("\nSupplementary figures, S1-S8:")
    for p in sorted(SUPPLEMENTARY_FIGURE_DIR.glob("*.png")):
        print(p.name)

    print("\nQC full-overview figures:")
    for p in sorted(QC_FULL_SUPPLEMENTARY_DIR.glob("*.png")):
        print(p.name)

    print("\nFigure legends:")
    print(legend_txt)
    print(legend_md)

    print("\nSingle-panel status:")
    print(df_panels.to_string(index=False))

    print("=" * 90)


if __name__ == "__main__":
    main()
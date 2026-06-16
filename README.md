# periodontal-mechanobiology-loading-code
Code accompanying the manuscript: A poroviscoelastic and mechanobiological framework for micro-dynamic orthodontic loading design

# Periodontal mechanobiology loading code

This repository contains the analysis and figure-generation code accompanying the manuscript:

**A poroviscoelastic and mechanobiological framework for micro-dynamic orthodontic loading design**

## Overview

This code implements a dry-lab computational framework for comparing static, intermittent, and micro-dynamic loading schedules in a periodontal mechanobiology context. The framework combines simplified poroviscoelastic model comparisons, dimensionless frequency-window analysis, waveform screening, Pareto-style design evaluation, and clinical-timescale bridging.

The repository is intended to support transparency, reproducibility, and post-publication reuse of the computational workflow reported in the manuscript. It does not contain patient-identifiable information, proprietary appliance designs, or patent-sensitive implementation details.

## Repository structure

* `code/`
  Main scripts used for model comparison, frequency-window analysis, waveform screening, Pareto evaluation, clinical-timescale bridging, and figure generation.

* `example_data/`
  Minimal example input tables required to reproduce representative outputs. These files are provided only for computational reproducibility and do not contain identifiable human data.

* `outputs/`
  Example output figures and summary tables generated from the scripts.

* `requirements.txt`
  Python package requirements.

* `R_session_info.txt`
  R package and session information for rheology-related plotting, if applicable.

* `CITATION.cff`
  Citation metadata for GitHub.

* `.zenodo.json`
  Metadata used by Zenodo when archiving GitHub releases.

## Main analyses

The code is organized around five computational components:

1. Viscoelastic model comparison
   Comparison of Kelvin-Voigt, Maxwell, standard linear solid, Burgers, and Prony-type representations under simplified orthodontic loading assumptions.

2. Frequency-window analysis
   Identification of a finite dimensionless loading window based on the relationship between loading period and tissue-level viscoelastic relaxation.

3. Waveform screening
   Comparison of sinusoidal, square, pulse, and related waveforms under energy-normalized conditions.

4. Design-space evaluation
   Pareto-style assessment of efficiency, robustness, and overload-related penalties.

5. Clinical-timescale bridging
   Comparison between micro-dynamic loading periods and conventional clinical renewal schedules.

## Reproducibility

To reproduce the main computational figures:

1. Install Python dependencies listed in `requirements.txt`.

2. Run scripts in numerical order from the `code/` folder.

3. Output files will be written to the `outputs/` folder.

Example:

`python code/part1_viscoelastic_models.py`

`python code/part2_frequency_window.py`

`python code/part3_waveform_screening.py`

`python code/part4_pareto_design.py`

`python code/part5_clinical_bridge.py`

For R-based rheology plotting:

`Rscript code/figure6_rheology_plot.R`

## Data availability

Only minimal example data required for reproducing the computational workflow are included. Raw experimental data, if any, should be described in the associated manuscript and made available according to ethical approval, institutional policy, and journal requirements.

## Code availability

The code is provided for academic review and research reuse. The archived version associated with the manuscript is available through Zenodo at:

DOI: to be added after Zenodo release

## License

This repository is released under the MIT License unless otherwise stated.

## Citation

If you use this code, please cite the associated manuscript and the archived Zenodo release.

Chen Zong. Periodontal mechanobiology loading code. Version 1.0.0. Zenodo. DOI: to be added after release.

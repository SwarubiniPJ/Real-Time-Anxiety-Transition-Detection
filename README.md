# ECG responses to anxiety-labelled video clips

This repository contains the analysis code for the ECG transition and
participant-independent state-classification study. The public code mirrors the
order of the final analysis notebook so that the generated results can be traced
to the manuscript without changing the statistical implementation.

## Repository layout

```text
.
├── analysis/
│   └── full_analysis.py       # Authoritative end-to-end analysis
├── notebooks/
│   └── ecg_analysis.ipynb     # Output-free readable notebook
├── config/
│   └── offsets.example.json   # Optional participant alignment offsets
├── data/
│   └── README.md              # Input format and access information
├── analysis_outputs/
│   └── README.md              # Description of generated artifacts
├── run_analysis.py            # Command-line launcher
├── requirements.txt           # Exact Python package versions
├── environment.yml            # Conda environment definition
└── .gitignore
```

Keep the manuscript's existing `figures/` and `tables/` directories alongside
these files. Do not duplicate manuscript figures inside the code directory.

## Analyses included

The pipeline performs:

1. ECG loading, filtering, R-peak detection, RR correction and quality control;
2. time-domain, frequency-domain, nonlinear and signal-level feature extraction;
3. prespecified 60-s transition comparisons with bootstrap confidence intervals,
   sign-flip tests and max-*T* multiplicity correction;
4. window-duration sensitivity, settling-latency and switch-heterogeneity analyses;
5. leakage-controlled leave-one-participant-out state models;
6. protocol-time, clip-identity, signal-quality and double-holdout controls;
7. direction-specific transition classifiers and structured circular-shift nulls;
8. nested expert-feature and tsfresh models; and
9. coefficient-stability and reproducibility exports.

## Installation

The analysis was developed with Python 3.11. Create an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, use Conda:

```bash
conda env create -f environment.yml
conda activate anxiety-ecg
```

On Apple silicon, the optional XGBoost analysis may additionally require
OpenMP (`brew install libomp`). The prespecified primary model does not depend
on XGBoost.

## Running the analysis

Run the complete manuscript analysis with:

```bash
python run_analysis.py \
  --data-dir /absolute/path/to/Data \
  --output-dir analysis_outputs
```

If recording alignment offsets are required, copy
`config/offsets.example.json`, enter the verified offsets and add:

```bash
  --offsets-json config/offsets.json
```

For a short installation and data-format check:

```bash
python run_analysis.py --data-dir /absolute/path/to/Data --quick
```

The `--quick` option reduces resampling and skips secondary models. It is only a
smoke test and must not be used to regenerate values reported in the manuscript.

## Reproducing the manuscript results

- Use the full command without `--quick`.
- Use the same 19 source recordings and any verified alignment offsets.
- Do not modify the fixed clip sequence, feature sets, random seed or analysis
  thresholds in `analysis/full_analysis.py`.
- Compare the newly generated CSV and figure files with the manuscript versions
  before replacing tracked files.

The raw data are not included. Public availability should follow the study's
consent, ethics and institutional data-governance requirements.

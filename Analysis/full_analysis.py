"""Complete ECG analysis pipeline exported from the thesis notebook.

Run this file through the repository-level ``run_analysis.py`` launcher.
The statistical implementation is intentionally kept in notebook order to
preserve correspondence with the analyses reported in the manuscript.
"""

import os
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value)


# %% Notebook code cell 3
import os, re, glob, json, sys, platform, warnings, itertools, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, welch
from scipy.ndimage import uniform_filter1d
from scipy.stats import skew as _skew, kurtosis as _kurt, ttest_1samp, wilcoxon
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import neurokit2 as nk
import antropy as ant
import pywt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, roc_curve, precision_recall_curve,
                             brier_score_loss)
from sklearn.inspection import permutation_importance
try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except Exception as exc:
    # The confirmatory model does not depend on XGBoost. On macOS, a missing libomp
    # should therefore disable only this optional secondary candidate, not the analysis.
    XGBClassifier = None
    HAVE_XGBOOST = False
    print("Optional XGBoost candidate disabled:", exc)
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

# Keep numerical/convergence warnings visible; they are part of the analysis audit trail.
warnings.filterwarnings("default")
SEED = 20250308
rng_global = np.random.default_rng(SEED)
TRAPZ = getattr(np, "trapezoid", np.trapz)
sns.set_theme(style="whitegrid", context="notebook")

print(sys.version)
print("platform:", platform.platform())
for package in [np, pd, nk, ant, pywt]:
    print(package.__name__, getattr(package, "__version__", "unknown"))

# %% Notebook code cell 5
# ---------------- LOCAL MAC SETTINGS ----------------
DATA_DIR = os.environ.get("ANXIETY_ECG_DATA_DIR")
if not DATA_DIR:
    raise RuntimeError("Set ANXIETY_ECG_DATA_DIR or use run_analysis.py --data-dir.")
OUTPUT_DIR = os.environ.get(
    "ANXIETY_ECG_OUTPUT_DIR",
    str(Path(DATA_DIR).parent / "analysis_outputs_v3"),
)
FS = 500
ECG_COL = 0
MAT_KEY = "data"
OFFSETS_PATH = os.environ.get("ANXIETY_ECG_OFFSETS_JSON")
if OFFSETS_PATH:
    with open(OFFSETS_PATH, encoding="utf-8") as offsets_file:
        OFFSETS = {str(k): float(v) for k, v in json.load(offsets_file).items()}
else:
    OFFSETS = {}  # seconds added to the reconstructed clip timeline

# ---------------- SIGNAL AND WINDOW SETTINGS ----------------
ECG_BAND = (0.5, 20.0)
FILTER_ORDER = 2
WINDOW_S, STEP_S, GUARD_S, PURITY = 60.0, 30.0, 15.0, 0.80
MIN_BEATS_STATE = 20
MIN_BEATS_TRANSITION = 3
MAX_CORRECTION_FRACTION = 0.20
MAX_GAP_S = 3.0
MIN_SUBJECT_NN = 100
RECORDING_TOLERANCE_S = 2.0

# ---------------- CONFIRMATORY ANALYSIS ----------------
PRIMARY_W = 60
PRIMARY_FEATURES = ["MHR", "SDNN", "RMSSD", "pNN50"]
SWEEP_WINDOWS = [5, 10, 15, 20, 30, 45, 60]
N_SIGNFLIP = int(os.environ.get("ANXIETY_ECG_N_SIGNFLIP", "20000"))
N_BOOT = int(os.environ.get("ANXIETY_ECG_N_BOOT", "5000"))
N_TRANSITION_PERM = int(os.environ.get("ANXIETY_ECG_N_TRANSITION_PERM", "1000"))
# Populated automatically from the QC table after the recordings are processed.
ANALYSIS_VERSION = "thesis_methodology_v3_2026-09-02"

# ---------------- LATENCY ----------------
PRE, HORIZON, TARGET_W = 30, 60, 20
GRID_HZ, SMOOTH_S, MIN_DELTA, PERSIST_S = 2.0, 5.0, 1.0, 5.0
LATENCY_FRACTION = 0.63

# ---------------- MODELLING ----------------
CORE_FEATS = ["MHR", "SDNN", "RMSSD", "pNN50"]
FULL_FEATS = ["MeanNN", "MHR", "SDHR", "SDNN", "RMSSD", "pNN50", "LF", "HF", "LFHF",
              "SampEn", "DFA_a1", "ecg_skew", "ecg_kurt", "ecg_ampstd"]
TRANSITION_FEATS = ["MHR", "dMHR_pp", "dSDNN_pp", "dRMSSD_pp",
                    "wav_energy", "wav_center", "wav_peakoff"]
TSFRESH_K = 20
TRANSITION_NEG_STEP_S = 60.0  # non-overlapping stable windows
RUN_TSFRESH = os.environ.get("ANXIETY_ECG_RUN_TSFRESH", "1") == "1"
RUN_SECONDARY_MODELS = os.environ.get("ANXIETY_ECG_RUN_SECONDARY_MODELS", "1") == "1"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Corrected fixed clip order from the thesis.
CLIPS = [(1,1,"amputation",3,16), (2,0,"music",1,59), (3,1,"orphan",3,38),
         (4,0,"happy",1,37), (5,1,"car crashes",5,17), (6,0,"Minions",3,51),
         (7,1,"natural disasters",14,42), (8,0,"nature",7,0)]
VIDEO_TOTAL_S = sum(m*60+s for _,_,_,m,s in CLIPS)

def build_timeline(offset=0.0):
    rows, t = [], float(offset)
    for clip, ai, desc, minutes, seconds in CLIPS:
        duration = minutes*60 + seconds
        rows.append({"clip":clip, "ai":ai, "desc":desc, "start":t, "end":t+duration})
        t += duration
    return pd.DataFrame(rows)

def switch_list(offset=0.0):
    tl, out = build_timeline(offset), []
    for i in range(len(tl)-1):
        a, b = tl.iloc[i], tl.iloc[i+1]
        direction = "AI->NAI" if a["ai"] == 1 else "NAI->AI"
        out.append({"sw":i+1, "t":float(a["end"]), "dir":direction,
                    "from":int(a["clip"]), "to":int(b["clip"]),
                    "ai_clip":int(a["clip"] if direction == "AI->NAI" else b["clip"]),
                    "ai_desc":a["desc"] if direction == "AI->NAI" else b["desc"]})
    return out

print(build_timeline().to_string(index=False))
print(pd.DataFrame(switch_list()).to_string(index=False))

# %% Notebook code cell 7
def sid_of(path):
    match = re.findall(r"A\d+", os.path.basename(path))
    return match[-1] if match else Path(path).stem

def bandpass(ecg, fs=FS):
    b, a = butter(FILTER_ORDER, ECG_BAND, btype="band", fs=fs)
    return filtfilt(b, a, np.asarray(ecg, dtype=float))

def detect_and_correct_rpeaks(ecg_filtered, fs=FS):
    _, info = nk.ecg_peaks(ecg_filtered, sampling_rate=fs, method="neurokit")
    raw_peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    corrected = raw_peaks.copy()
    correction_info = {}
    try:
        correction_info, corrected = nk.signal_fixpeaks(
            raw_peaks, sampling_rate=fs, iterative=True, method="Kubios", show=False)
        corrected = np.asarray(corrected, dtype=int)
    except Exception as exc:
        print("signal_fixpeaks warning:", exc)
    corrected = np.unique(corrected[(corrected > 0) & (corrected < len(ecg_filtered))])
    return raw_peaks, corrected, correction_info

def one_to_one_peak_matches(a, b, tolerance_samples):
    # Number of one-to-one matches between two ordered peak arrays.
    a, b = np.asarray(a, int), np.asarray(b, int)
    i = j = matched = 0
    while i < len(a) and j < len(b):
        delta = a[i] - b[j]
        if abs(delta) <= tolerance_samples:
            matched += 1; i += 1; j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return matched

def clean_nn_from_peaks(peaks, fs=FS, hampel_half_width=5):
    # Return interpolated NN values, acquisition times and an interval-level correction mask.
    peaks = np.asarray(peaks, int)
    rr = np.diff(peaks) / fs * 1000.0
    times = peaks[1:] / fs
    physiologic = (rr >= 300.0) & (rr <= 2000.0)

    series = pd.Series(rr)
    width = 2 * hampel_half_width + 1
    local_median = series.rolling(width, center=True, min_periods=5).median().to_numpy()
    abs_dev = np.abs(rr - local_median)
    local_mad = pd.Series(abs_dev).rolling(width, center=True, min_periods=5).median().to_numpy()
    # The relative floor prevents tiny local MAD values from over-correcting normal sinus variability.
    threshold = np.maximum(3.0 * 1.4826 * local_mad, 0.20 * np.abs(local_median))
    hampel_bad = np.isfinite(threshold) & (abs_dev > threshold)
    corrected_mask = (~physiologic) | hampel_bad | (~np.isfinite(rr))

    valid = ~corrected_mask
    if valid.sum() < 3:
        raise ValueError("Fewer than three valid RR intervals after quality screening")
    clean = rr.copy()
    clean[corrected_mask] = np.interp(times[corrected_mask], times[valid], rr[valid])
    return clean, times, corrected_mask

def subject_quality(raw_peaks, corrected_peaks, rr_corrected, duration, required_duration):
    tolerance = int(round(0.05 * FS))
    matched = one_to_one_peak_matches(raw_peaks, corrected_peaks, tolerance)
    peak_edits = (len(raw_peaks) - matched) + (len(corrected_peaks) - matched)
    peak_correction_fraction = peak_edits / max(len(raw_peaks), 1)
    rr_correction_fraction = float(np.mean(rr_corrected)) if len(rr_corrected) else np.nan
    recording_complete = duration >= required_duration - RECORDING_TOLERANCE_S
    return {
        "raw_rpeaks": len(raw_peaks),
        "corrected_rpeaks": len(corrected_peaks),
        "peak_edits": peak_edits,
        "peak_correction_fraction": peak_correction_fraction,
        "rr_intervals": len(rr_corrected),
        "rr_correction_fraction": rr_correction_fraction,
        "recording_s": duration,
        "required_recording_s": required_duration,
        "recording_shortfall_s": max(0.0, required_duration-duration),
        "recording_complete": recording_complete,
        "descriptive_qc_flag": (
            (not recording_complete) or len(rr_corrected) < MIN_SUBJECT_NN
            or peak_correction_fraction > MAX_CORRECTION_FRACTION
        )
    }

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.mat")))
assert paths, f"No .mat files found in {DATA_DIR}"
subjects = {sid_of(p): p for p in paths}
assert len(subjects) == len(paths), "Duplicate subject identifiers detected"

CACHE, qc_rows = {}, []
for sid, path in subjects.items():
    mat = sio.loadmat(path)
    if MAT_KEY not in mat:
        raise KeyError(f"{path}: missing MATLAB variable '{MAT_KEY}'")
    raw = np.asarray(mat[MAT_KEY])
    if raw.ndim != 2 or ECG_COL >= raw.shape[1]:
        raise ValueError(f"{path}: expected 2-D data and ECG_COL={ECG_COL}; got {raw.shape}")
    ecg = raw[:, ECG_COL].astype(float)
    if not np.isfinite(ecg).all():
        raise ValueError(f"{path}: ECG contains NaN/Inf")

    ef = bandpass(ecg)
    raw_peaks, peaks, fix_info = detect_and_correct_rpeaks(ef)
    nn, nn_t, nn_corrected = clean_nn_from_peaks(peaks)
    required_duration = VIDEO_TOTAL_S + max(0.0, float(OFFSETS.get(sid, 0.0)))
    q = subject_quality(raw_peaks, peaks, nn_corrected, len(ecg)/FS, required_duration)
    q["subject"] = sid
    qc_rows.append(q)
    CACHE[sid] = {
        "ef": ef, "peaks": peaks, "nn": nn, "t": nn_t,
        "nn_corrected": nn_corrected, "hr": 60000.0/nn,
        "total": len(ecg)/FS, "qc": q, "fix_info": fix_info
    }

qc = pd.DataFrame(qc_rows).sort_values("subject")
qc.to_csv(Path(OUTPUT_DIR)/"Table_S2_subject_quality_control.csv", index=False)
display(qc)
print(f"All {len(subjects)} participants are retained. QC flags are descriptive; failed windows are removed.")

# %% Notebook code cell 8
def alignment_diagnostic(sid):
    d, tl = CACHE[sid], build_timeline(OFFSETS.get(sid, 0.0))
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.plot(d["t"], d["hr"], lw=.45, color="black", alpha=.65)
    for _, row in tl.iterrows():
        ax.axvspan(row["start"], row["end"], color="#d1495b" if row["ai"] else "#2e86ab", alpha=.12)
    ax.set(title=f"{sid}: reconstructed alignment; surplus={d['total']-VIDEO_TOTAL_S:.1f} s",
           xlabel="Time (s)", ylabel="Instantaneous HR (bpm)")
    plt.tight_layout(); plt.show()

for sid in list(subjects)[:3]:
    alignment_diagnostic(sid)

# %% Notebook code cell 10
def f_time(nn):
    nn = np.asarray(nn, float)
    out = {k:np.nan for k in ["MeanNN","MHR","SDHR","SDNN","RMSSD","pNN50"]}
    if len(nn) < 3: return out
    hr, dnn = 60000.0/nn, np.diff(nn)
    out.update({"MeanNN":float(np.mean(nn)),
                "MHR":float(60000.0/np.mean(nn)),       # one definition used everywhere
                "SDHR":float(np.std(hr, ddof=1)), "SDNN":float(np.std(nn, ddof=1)),
                "RMSSD":float(np.sqrt(np.mean(dnn**2))),
                "pNN50":float(100*np.mean(np.abs(dnn)>50))})
    return out

def f_freq(nn, t):
    out = {"LF":np.nan, "HF":np.nan, "LFHF":np.nan}
    nn, t = np.asarray(nn,float), np.asarray(t,float)
    if len(nn) < 20 or len(t) != len(nn) or t[-1]-t[0] < 30: return out
    if np.max(np.diff(t)) > MAX_GAP_S: return out
    grid = np.arange(t[0], t[-1], 0.25)
    if len(grid) < 120: return out
    xi = np.interp(grid, t, nn); xi -= np.mean(xi)
    freq, power = welch(xi, fs=4.0, nperseg=min(256, len(xi)))
    lm = (freq>=0.04)&(freq<0.15); hm = (freq>=0.15)&(freq<0.40)
    lf, hf = TRAPZ(power[lm],freq[lm]), TRAPZ(power[hm],freq[hm])
    out.update({"LF":float(lf), "HF":float(hf), "LFHF":float(lf/hf) if hf>0 else np.nan})
    return out

def f_nonlinear(nn):
    out = {"SampEn":np.nan, "DFA_a1":np.nan}
    nn = np.asarray(nn,float)
    # Retained for completeness, but considered secondary for 60-s windows.
    if len(nn) >= 40:
        try: out["SampEn"] = float(ant.sample_entropy(nn))
        except Exception: pass
        try: out["DFA_a1"] = float(ant.detrended_fluctuation(nn))
        except Exception: pass
    return out

def f_morph(ecg_segment):
    out = {"ecg_skew":np.nan, "ecg_kurt":np.nan, "ecg_ampstd":np.nan}
    if len(ecg_segment)>10:
        out.update({"ecg_skew":float(_skew(ecg_segment)),
                    "ecg_kurt":float(_kurt(ecg_segment)),
                    "ecg_ampstd":float(np.std(ecg_segment))})
    return out

def f_wavelet(hr_series):
    out = {"wav_energy":np.nan,"wav_center":np.nan,"wav_peakoff":np.nan}
    if len(hr_series)>=16:
        try:
            coef,_ = pywt.cwt(hr_series-np.mean(hr_series),np.arange(2,16),"morl")
            energy = np.sum(coef**2,axis=0); total = energy.sum()
            if total>0:
                pos=np.arange(len(energy))/max(len(energy)-1,1)
                out.update({"wav_energy":float(total),"wav_center":float(np.sum(pos*energy)/total),
                            "wav_peakoff":float(abs(np.argmax(energy)/max(len(energy)-1,1)-.5))})
        except Exception: pass
    return out

def win_change(nn,t,ws,we):
    mid=ws+(we-ws)/2; pre=nn[t<mid]; post=nn[t>=mid]
    out={"dMHR_pp":np.nan,"dSDNN_pp":np.nan,"dRMSSD_pp":np.nan}
    if len(pre)>=5 and len(post)>=5:
        out["dMHR_pp"]=60000/np.mean(post)-60000/np.mean(pre)
        out["dSDNN_pp"]=np.std(post,ddof=1)-np.std(pre,ddof=1)
        out["dRMSSD_pp"]=np.sqrt(np.mean(np.diff(post)**2))-np.sqrt(np.mean(np.diff(pre)**2))
    return out


def feature_window(sid, ws, we, include_transition=False, min_beats=None):
    d = CACHE[sid]
    mask = (d["t"] >= ws) & (d["t"] < we)
    nn, t = d["nn"][mask], d["t"][mask]
    corrected = d["nn_corrected"][mask]
    required = MIN_BEATS_STATE if min_beats is None else int(min_beats)
    correction_fraction = float(np.mean(corrected)) if len(corrected) else np.nan
    if len(nn) < required:
        return None
    if not np.isfinite(correction_fraction) or correction_fraction > MAX_CORRECTION_FRACTION:
        return None
    if len(t) > 1 and np.max(np.diff(t)) > MAX_GAP_S:
        return None

    row = {
        "window_n_beats": len(nn),
        "window_rr_correction_fraction": correction_fraction
    }
    row.update(f_time(nn)); row.update(f_freq(nn, t)); row.update(f_nonlinear(nn))
    row.update(f_morph(d["ef"][max(0, int(ws*FS)):min(len(d["ef"]), int(we*FS))]))
    if include_transition:
        row.update(win_change(nn, t, ws, we))
        grid = np.arange(ws, we, .25)
        row.update(f_wavelet(np.interp(grid, t, 60000/nn) if len(nn) > 1 else np.array([])))
    return row

# %% Notebook code cell 13
# ============================================================
# CONFIRMATORY SUBJECT-LEVEL TRANSITION CALCULATIONS
# Run after CACHE, feature_window, and switch_list are defined
# ============================================================

def bootstrap_ci(x, n=N_BOOT, seed=SEED, confidence=0.95):
    """Percentile bootstrap CI for the subject-level mean."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)

    bootstrap_means = rng.choice(
        x,
        size=(int(n), len(x)),
        replace=True
    ).mean(axis=1)

    alpha = (1.0 - confidence) / 2.0

    return (
        float(np.quantile(bootstrap_means, alpha)),
        float(np.quantile(bootstrap_means, 1.0 - alpha))
    )


def signflip_p(x, n_perm=N_SIGNFLIP, seed=SEED):
    """Two-sided Monte Carlo sign-flip test of a zero mean change."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    observed = abs(np.mean(x))
    rng = np.random.default_rng(seed)

    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(int(n_perm), len(x)),
        replace=True
    )

    permuted_statistics = np.abs(
        np.mean(signs * x[None, :], axis=1)
    )

    # Add-one correction for a Monte Carlo permutation test
    return float(
        (np.sum(permuted_statistics >= observed) + 1)
        / (int(n_perm) + 1)
    )


def subject_switch_differences(window_s, direction):
    """
    Calculate post-minus-pre changes around each transition and then
    average repeated transitions within each participant.

    The returned rows are statistically independent participant-level
    summaries, rather than pooled transition observations.
    """
    if direction not in {"NAI->AI", "AI->NAI"}:
        raise ValueError(
            "direction must be either 'NAI->AI' or 'AI->NAI'"
        )

    window_s = float(window_s)

    if window_s <= 0:
        raise ValueError("window_s must be positive")

    subject_rows = []

    for sid in sorted(CACHE):
        switch_rows = []

        for transition in switch_list(
            OFFSETS.get(sid, 0.0)
        ):
            if transition["dir"] != direction:
                continue

            transition_time = float(transition["t"])

            # Non-overlapping windows immediately before and after
            # the annotated transition.
            pre = feature_window(
                sid,
                transition_time - window_s,
                transition_time,
                min_beats=MIN_BEATS_TRANSITION
            )

            post = feature_window(
                sid,
                transition_time,
                transition_time + window_s,
                min_beats=MIN_BEATS_TRANSITION
            )

            # Both sides must pass the prespecified window-level QC.
            if pre is None or post is None:
                continue

            row = {
                "subject": sid,
                "switch": int(transition["sw"])
            }

            for feature in PRIMARY_FEATURES:
                pre_value = pre.get(feature, np.nan)
                post_value = post.get(feature, np.nan)

                if (
                    np.isfinite(pre_value)
                    and np.isfinite(post_value)
                ):
                    row[feature] = post_value - pre_value
                else:
                    row[feature] = np.nan

            switch_rows.append(row)

        # Average repeated switches within the participant so that
        # each participant contributes at most one value per endpoint.
        if switch_rows:
            switch_frame = pd.DataFrame(switch_rows)

            subject_row = {
                "subject": sid,
                "direction": direction,
                "n_switches": len(switch_frame)
            }

            for feature in PRIMARY_FEATURES:
                values = switch_frame[feature].to_numpy(dtype=float)
                values = values[np.isfinite(values)]

                subject_row[feature] = (
                    float(np.mean(values))
                    if len(values)
                    else np.nan
                )

                subject_row[f"n_switches_{feature}"] = len(values)

            subject_rows.append(subject_row)

    expected_columns = (
        ["subject", "direction", "n_switches"]
        + PRIMARY_FEATURES
        + [f"n_switches_{f}" for f in PRIMARY_FEATURES]
    )

    return pd.DataFrame(
        subject_rows,
        columns=expected_columns
    )


# ------------------------------------------------------------
# PRESPECIFIED 60-S PRIMARY ANALYSIS
# ------------------------------------------------------------

primary_differences = {
    direction: subject_switch_differences(
        PRIMARY_W,
        direction
    )
    for direction in ["NAI->AI", "AI->NAI"]
}

# This is the object required by the Nature-style Figure 2 cell.
primary_vectors = {}

primary_rows = []

for direction, differences in primary_differences.items():

    for feature in PRIMARY_FEATURES:
        values = (
            differences[feature]
            .dropna()
            .to_numpy(dtype=float)
        )

        primary_vectors[(direction, feature)] = values

        ci_lo, ci_hi = bootstrap_ci(
            values,
            n=N_BOOT,
            seed=SEED
        )

        primary_rows.append({
            "direction": direction,
            "window_s": PRIMARY_W,
            "feature": feature,
            "n": len(values),
            "mean_change": (
                float(np.mean(values))
                if len(values)
                else np.nan
            ),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_signflip": signflip_p(
                values,
                n_perm=N_SIGNFLIP,
                seed=SEED
            )
        })

primary_results = pd.DataFrame(primary_rows)

# BH values are useful descriptively here; retain your prespecified
# primary multiplicity procedure separately if it is max-T.
valid = primary_results["p_signflip"].notna()

primary_results.loc[
    valid,
    "q_BH"
] = multipletests(
    primary_results.loc[valid, "p_signflip"],
    method="fdr_bh"
)[1]

primary_results.to_csv(
    Path(OUTPUT_DIR) / "Table_primary_transition_results.csv",
    index=False
)

display(primary_results.round(4))

for direction, frame in primary_differences.items():
    print(
        direction,
        "participants:",
        frame["subject"].nunique()
    )

# %% Notebook code cell 14
# ============================================================
# NATURE-STYLE PRIMARY TRANSITION FIGURE
# Run after primary_vectors has been created
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ------------------------------------------------------------
# COLOURS AND LABELS
# ------------------------------------------------------------

# Colour-blind-friendly palette
DIRECTION_STYLE = {
    "NAI->AI": {
        "color": "#009E73",       # teal-green
        "x": 0,
        "xlabel": "Entering"
    },
    "AI->NAI": {
        "color": "#E69F00",       # amber-orange
        "x": 1,
        "xlabel": "Leaving"
    }
}

FEATURE_TITLES = {
    "MHR": "MHR",
    "SDNN": "SDNN",
    "RMSSD": "RMSSD",
    "pNN50": "pNN50"
}

# Change is calculated as post-transition minus pre-transition
Y_LABELS = {
    "MHR": r"$\Delta$ MHR (beats min$^{-1}$)",
    "SDNN": r"$\Delta$ SDNN (ms)",
    "RMSSD": r"$\Delta$ RMSSD (ms)",
    "pNN50": r"$\Delta$ pNN50 (percentage points)"
}

PANEL_LABELS = ["a", "b", "c", "d"]


# ------------------------------------------------------------
# GLOBAL FIGURE STYLE
# ------------------------------------------------------------

nature_rc = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 22,
    "font.weight": "bold",

    "axes.labelsize": 22,
    "axes.labelweight": "bold",
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.linewidth": 2.8,

    "xtick.labelsize": 22,
    "ytick.labelsize": 22,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    # Preserve editable text in vector files
    "pdf.fonttype": 42,
    "ps.fonttype": 42
}


# ------------------------------------------------------------
# LEGEND ENTRIES
# ------------------------------------------------------------

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=15,
        markerfacecolor=DIRECTION_STYLE["NAI->AI"]["color"],
        markeredgecolor="white",
        markeredgewidth=1.5,
        label=r"NAI $\rightarrow$ AI"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=15,
        markerfacecolor=DIRECTION_STYLE["AI->NAI"]["color"],
        markeredgecolor="white",
        markeredgewidth=1.5,
        label=r"AI $\rightarrow$ NAI"
    )
]


# ------------------------------------------------------------
# CREATE FIGURE
# ------------------------------------------------------------

with plt.rc_context(nature_rc):

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(18, 14),
        sharex=True,
        constrained_layout=False
    )

    # Use consistent horizontal jitter across all four panels
    jitter_values = {}

    for j, direction in enumerate(["NAI->AI", "AI->NAI"]):

        n_points = len(
            primary_vectors[
                (direction, PRIMARY_FEATURES[0])
            ]
        )

        jitter_values[direction] = np.random.default_rng(
            SEED + j
        ).normal(
            loc=0,
            scale=0.055,
            size=n_points
        )

    # --------------------------------------------------------
    # PLOT EACH FEATURE
    # --------------------------------------------------------

    for panel_label, ax, feature in zip(
        PANEL_LABELS,
        axes.ravel(),
        PRIMARY_FEATURES
    ):

        for direction in ["NAI->AI", "AI->NAI"]:

            style = DIRECTION_STYLE[direction]

            x_position = style["x"]
            point_color = style["color"]

            values = np.asarray(
                primary_vectors[(direction, feature)],
                dtype=float
            )

            # Remove any non-finite values
            values = values[np.isfinite(values)]

            # Participant-level observations
            ax.scatter(
                np.full(len(values), x_position)
                + jitter_values[direction][:len(values)],
                values,
                s=105,
                color=point_color,
                edgecolor="white",
                linewidth=1.5,
                alpha=0.90,
                zorder=3
            )

            # Group mean and participant-bootstrap 95% CI
            ci_low, ci_high = bootstrap_ci(values)
            mean_value = np.mean(values)

            lower_error = mean_value - ci_low
            upper_error = ci_high - mean_value

            ax.errorbar(
                x_position,
                mean_value,
                yerr=[
                    [lower_error],
                    [upper_error]
                ],
                fmt="D",
                markersize=13,
                markerfacecolor="black",
                markeredgecolor="white",
                markeredgewidth=1.8,
                color="black",
                ecolor="black",
                elinewidth=4.5,
                capsize=11,
                capthick=4.5,
                zorder=5
            )

        # ----------------------------------------------------
        # PANEL FORMATTING
        # ----------------------------------------------------

        # Horizontal reference line for no change
        ax.axhline(
            y=0,
            color="black",
            linewidth=2.2,
            linestyle="-",
            zorder=1
        )

        # Remove all internal gridlines
        ax.grid(False)

        # Thicken the complete box surrounding each panel
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(2.8)

        # X-axis formatting
        ax.set_xlim(-0.35, 1.35)

        ax.set_xticks([0, 1])

        ax.set_xticklabels(
            ["Entering", "Leaving"],
            fontsize=22,
            fontweight="bold"
        )

        # Ensure x-axis labels appear on all four panels
        ax.tick_params(
            axis="x",
            labelbottom=True
        )

        # Feature-specific y-axis label
        ax.set_ylabel(
            Y_LABELS[feature],
            fontsize=22,
            fontweight="bold",
            labelpad=12
        )

        # Feature title
        ax.set_title(
            FEATURE_TITLES[feature],
            fontsize=22,
            fontweight="bold",
            pad=15
        )

        # Tick appearance
        ax.tick_params(
            axis="both",
            which="major",
            width=2.8,
            length=8,
            direction="out",
            pad=8
        )

        # Make all tick labels bold
        for tick_label in (
            ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            tick_label.set_fontsize(22)
            tick_label.set_fontweight("bold")

        # Add additional space at the top of the panel
        # so that the legend does not cover observations
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        if y_range > 0:
            ax.set_ylim(
                y_min,
                y_max + (0.32 * y_range)
            )

        # Nature-style panel label
        ax.text(
            -0.16,
            1.09,
            panel_label,
            transform=ax.transAxes,
            fontsize=26,
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False
        )

        # ----------------------------------------------------
        # BOXED LEGEND INSIDE EACH SUBPLOT
        # ----------------------------------------------------

        panel_legend = ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.025, 0.975),
            ncol=1,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
            prop={
                "family": "sans-serif",
                "size": 18,
                "weight": "bold"
            },
            borderpad=0.55,
            labelspacing=0.35,
            handletextpad=0.45,
            handlelength=1.0
        )

        # Thicken the legend box
        panel_legend.get_frame().set_linewidth(2.2)
        panel_legend.get_frame().set_edgecolor("black")

    # --------------------------------------------------------
    # FIGURE SPACING
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.08,
        top=0.94,
        wspace=0.30,
        hspace=0.35
    )

    # --------------------------------------------------------
    # SAVE FIGURE
    # --------------------------------------------------------

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    png_path = (
        Path(OUTPUT_DIR)
        / "Figure_2_primary_transition_effects_nature.png"
    )

    pdf_path = (
        Path(OUTPUT_DIR)
        / "Figure_2_primary_transition_effects_nature.pdf"
    )

    # High-resolution PNG
    fig.savefig(
        png_path,
        dpi=900,
        bbox_inches="tight",
        facecolor="white"
    )

    # Vector PDF for manuscript submission
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


print(f"PNG saved to: {png_path}")
print(f"Vector PDF saved to: {pdf_path}")

# %% Notebook code cell 15
# Full 56-cell window sweep is sensitivity analysis, not model/endpoint selection.
sweep_rows=[]
for direction in ["NAI->AI","AI->NAI"]:
    for w in SWEEP_WINDOWS:
        dd=subject_switch_differences(w,direction)
        for feat in PRIMARY_FEATURES:
            x=dd[feat].dropna().to_numpy(); lo,hi=bootstrap_ci(x,n=2000)
            sweep_rows.append({"direction":direction,"window_s":w,"feature":feat,"n":len(x),
                               "mean_change":np.mean(x) if len(x) else np.nan,"ci_lo":lo,"ci_hi":hi,
                               "p_signflip":signflip_p(x,n_perm=10000,seed=SEED+w)})
sweep_results=pd.DataFrame(sweep_rows)
ok=sweep_results["p_signflip"].notna()
sweep_results.loc[ok,"q_BH_sensitivity"]=multipletests(sweep_results.loc[ok,"p_signflip"],method="fdr_bh")[1]
sweep_results.to_csv(Path(OUTPUT_DIR)/"Table_S4_all_window_feature_tests.csv",index=False)

fig,ax=plt.subplots(figsize=(7,4.5))
for direction,color in [("NAI->AI","#d1495b"),("AI->NAI","#2e86ab")]:
    z=sweep_results[(sweep_results["direction"]==direction)&(sweep_results["feature"]=="MHR")]
    ax.plot(z["window_s"],z["mean_change"],"o-",color=color,label=direction)
    ax.fill_between(z["window_s"],z["ci_lo"],z["ci_hi"],color=color,alpha=.18)
ax.axhline(0,color="black",lw=1); ax.set(xlabel="Pre/post window (s)",ylabel="MHR change (bpm)",
    title="MHR transition sensitivity (subject-level 95% bootstrap CI)")
ax.legend(); plt.tight_layout(); plt.savefig(Path(OUTPUT_DIR)/"Figure_S4_window_sensitivity.png",dpi=300); plt.show()

# %% Notebook code cell 16
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# NATURE-STYLE WINDOW-SENSITIVITY FIGURE
# Run after sweep_results has been created
# ============================================================

required_columns = {
    "direction",
    "feature",
    "window_s",
    "mean_change",
    "ci_lo",
    "ci_hi"
}

missing_columns = required_columns.difference(sweep_results.columns)

if missing_columns:
    raise ValueError(
        f"sweep_results is missing columns: {sorted(missing_columns)}"
    )


# ------------------------------------------------------------
# COLOURS, MARKERS, AND LABELS
# ------------------------------------------------------------

DIRECTION_STYLE = {
    "NAI->AI": {
        "color": "#009E73",
        "marker": "o",
        "label": r"NAI $\rightarrow$ AI"
    },
    "AI->NAI": {
        "color": "#E69F00",
        "marker": "s",
        "label": r"AI $\rightarrow$ NAI"
    }
}


# ------------------------------------------------------------
# GLOBAL FIGURE STYLE
# ------------------------------------------------------------

nature_rc = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "DejaVu Sans"
    ],
    "font.size": 12,
    "font.weight": "bold",

    "axes.labelsize": 13,
    "axes.labelweight": "bold",

    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.linewidth": 2.0,

    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    # Preserve editable text in vector outputs
    "pdf.fonttype": 42,
    "ps.fonttype": 42
}


# ------------------------------------------------------------
# PREPARE EVENLY SPACED X-AXIS POSITIONS
# ------------------------------------------------------------

mhr_results = sweep_results.loc[
    sweep_results["feature"] == "MHR"
].copy()

if mhr_results.empty:
    raise ValueError(
        "No MHR rows were found in sweep_results."
    )

# Actual window durations shown as tick labels
window_labels = sorted(
    mhr_results["window_s"].dropna().unique()
)

if not window_labels:
    raise ValueError(
        "No valid window durations were found for MHR."
    )

# Categorical locations create equal spacing
x_positions = np.arange(
    len(window_labels),
    dtype=float
)


# ------------------------------------------------------------
# CREATE FIGURE
# ------------------------------------------------------------

with plt.rc_context(nature_rc):

    fig, ax = plt.subplots(
        figsize=(8.4, 5.5)
    )

    for direction in ["NAI->AI", "AI->NAI"]:

        style = DIRECTION_STYLE[direction]

        # Index and reindex ensure that both directions follow the
        # same window order.
        z = (
            mhr_results.loc[
                mhr_results["direction"] == direction,
                [
                    "window_s",
                    "mean_change",
                    "ci_lo",
                    "ci_hi"
                ]
            ]
            .drop_duplicates(
                subset="window_s",
                keep="first"
            )
            .set_index("window_s")
            .reindex(window_labels)
        )

        mean_change = z[
            "mean_change"
        ].to_numpy(dtype=float)

        ci_low = z[
            "ci_lo"
        ].to_numpy(dtype=float)

        ci_high = z[
            "ci_hi"
        ].to_numpy(dtype=float)

        # Bootstrap 95% confidence band
        ax.fill_between(
            x_positions,
            ci_low,
            ci_high,
            color=style["color"],
            alpha=0.16,
            linewidth=0,
            interpolate=True,
            zorder=1
        )

        # Mean sensitivity curve
        ax.plot(
            x_positions,
            mean_change,
            color=style["color"],
            linewidth=3.0,
            marker=style["marker"],
            markersize=9,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=1.5,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=style["label"],
            zorder=3
        )


    # --------------------------------------------------------
    # REFERENCE LINE
    # --------------------------------------------------------

    ax.axhline(
        y=0,
        color="black",
        linewidth=1.8,
        linestyle="-",
        zorder=2
    )


    # --------------------------------------------------------
    # AXIS FORMATTING
    # --------------------------------------------------------

    # Equally spaced tick locations
    ax.set_xticks(x_positions)

    # Show the real window durations
    ax.set_xticklabels(
        [f"{window:g}" for window in window_labels]
    )

    ax.set_xlim(
        x_positions[0] - 0.30,
        x_positions[-1] + 0.30
    )

    ax.set_xlabel(
        "Pre/post window duration (s)",
        labelpad=10
    )

    ax.set_ylabel(
        r"$\Delta$ MHR (beats min$^{-1}$)",
        labelpad=10
    )

    ax.set_title(
        "MHR transition sensitivity",
        pad=14
    )

    # Add vertical breathing room without changing the data
    ax.margins(y=0.12)


    # --------------------------------------------------------
    # PANEL APPEARANCE
    # --------------------------------------------------------

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.0)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=2.0,
        length=7,
        pad=6,
        colors="black"
    )

    for tick_label in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        tick_label.set_fontweight("bold")


    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linewidth=3.0,
            marker=style["marker"],
            markersize=9,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=style["label"]
        )
        for style in DIRECTION_STYLE.values()
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        prop={
            "family": "sans-serif",
            "size": 11,
            "weight": "bold"
        },
        borderpad=0.7,
        labelspacing=0.6,
        handlelength=2.4
    )

    legend.get_frame().set_linewidth(1.7)


    # --------------------------------------------------------
    # FIGURE NOTE
    # --------------------------------------------------------


    fig.subplots_adjust(
        left=0.15,
        right=0.97,
        bottom=0.19,
        top=0.90
    )


    # --------------------------------------------------------
    # SAVE FIGURE
    # --------------------------------------------------------

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    png_path = (
        output_path
        / "Figure_S4_window_sensitivity_nature.png"
    )

    pdf_path = (
        output_path
        / "Figure_S4_window_sensitivity_nature.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


print(f"PNG saved to: {png_path}")
print(f"PDF saved to: {pdf_path}")

# %% Notebook code cell 18
def hr_trajectory(sid,center,pre=PRE,horizon=HORIZON,grid_hz=GRID_HZ,smooth_s=SMOOTH_S):
    d=CACHE[sid]; pad=max(smooth_s,2.0)
    if center-pre-pad<d["t"][0] or center+horizon+pad>d["t"][-1]: return None,None
    quality=(d["t"]>=center-pre-pad)&(d["t"]<center+horizon+pad)
    if quality.sum()<MIN_BEATS_STATE or np.mean(d["nn_corrected"][quality])>MAX_CORRECTION_FRACTION:
        return None,None
    expanded=np.arange(center-pre-pad,center+horizon+pad,1/grid_hz)
    h=np.interp(expanded,d["t"],d["hr"])
    size=max(1,int(round(smooth_s*grid_hz)))
    h=uniform_filter1d(h,size=size,mode="reflect")
    keep=(expanded>=center-pre)&(expanded<center+horizon)
    return expanded[keep],h[keep]

def estimate_latency(tu,h,center,fraction=LATENCY_FRACTION,min_delta=MIN_DELTA,
                     persist_s=PERSIST_S,pre=PRE,horizon=HORIZON,target_w=TARGET_W,
                     grid_hz=GRID_HZ):
    out={"base":np.nan,"target":np.nan,"delta":np.nan,"latency":np.nan}
    if tu is None:return out
    base=np.mean(h[(tu>=center-pre)&(tu<center)])
    target=np.mean(h[(tu>=center+horizon-target_w)&(tu<center+horizon)])
    delta=target-base; out.update({"base":base,"target":target,"delta":delta})
    if not np.isfinite(delta) or abs(delta)<min_delta:return out
    post=h[tu>=center]; pt=tu[tu>=center]-center
    crossed=post>=base+fraction*delta if delta>0 else post<=base+fraction*delta
    hold=max(1,int(round(persist_s*grid_hz)))
    for i in range(0,len(post)-hold+1):
        if np.all(crossed[i:i+hold]):out["latency"]=float(pt[i]);break
    return out

# Unit test: a constant trajectory must remain constant after smoothing.
test=np.full(400,75.0); sm=uniform_filter1d(test,size=11,mode="reflect")
assert np.max(np.abs(sm-test))<1e-12, "Latency smoother creates an edge artefact"

lat_rows=[]
for sid in subjects:
    for sw in switch_list(OFFSETS.get(sid,0.0)):
        tu,h=hr_trajectory(sid,sw["t"]); est=estimate_latency(tu,h,sw["t"])
        lat_rows.append({"subject":sid,**sw,**est})
latency=pd.DataFrame(lat_rows)
latency.to_csv(Path(OUTPUT_DIR)/"Table_S5_subject_switch_latency.csv",index=False)

# ============================================================
# NATURE-STYLE SUBJECT-BY-SWITCH HEATMAP
# ============================================================

from matplotlib.lines import Line2D

heatmap_rc = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "font.weight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.linewidth": 2.0,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
}


# ------------------------------------------------------------
# PREPARE HEATMAP DATA
# ------------------------------------------------------------

subject_order = sorted(
    latency["subject"].dropna().unique()
)

switch_order = sorted(
    latency["sw"].dropna().unique()
)

piv = (
    latency.pivot(
        index="subject",
        columns="sw",
        values="delta"
    )
    .reindex(
        index=subject_order,
        columns=switch_order
    )
)

finite_values = piv.to_numpy(dtype=float)
finite_values = finite_values[np.isfinite(finite_values)]

if len(finite_values) == 0:
    raise ValueError(
        "No finite MHR delta values are available for the heatmap."
    )

# Symmetric limits ensure that zero is the visual midpoint.
color_limit = np.max(np.abs(finite_values))

if color_limit == 0:
    color_limit = 1.0


# Direction associated with each switch
switch_directions = (
    latency[
        ["sw", "dir"]
    ]
    .drop_duplicates(subset="sw")
    .set_index("sw")["dir"]
    .to_dict()
)

direction_display = {
    "NAI->AI": "NAI → AI",
    "AI->NAI": "AI → NAI"
}

x_tick_labels = [
    (
        f"Switch {int(switch_number)}\n"
        f"{direction_display.get(switch_directions.get(switch_number), '')}"
    )
    for switch_number in switch_order
]


# ------------------------------------------------------------
# CREATE THE FIGURE
# ------------------------------------------------------------

with plt.rc_context(heatmap_rc):

    fig, ax = plt.subplots(
        figsize=(12.5, 8.5)
    )

    # Gray cells indicate unavailable/failed-QC observations.
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E6E6E6")

    heatmap = sns.heatmap(
        piv,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-color_limit,
        vmax=color_limit,
        mask=piv.isna(),
        linewidths=0.8,
        linecolor="white",
        annot=False,
        cbar=True,
        cbar_kws={
            "label": r"$\Delta$ MHR (beats min$^{-1}$)",
            "shrink": 0.82,
            "pad": 0.025,
            "aspect": 28
        }
    )


    # --------------------------------------------------------
    # AXIS LABELS AND TITLE
    # --------------------------------------------------------



    ax.set_xlabel(
        "Annotated film transition",
        fontsize=14,
        fontweight="bold",
        labelpad=14
    )

    ax.set_ylabel(
        "Participant",
        fontsize=14,
        fontweight="bold",
        labelpad=12
    )

    ax.set_xticklabels(
        x_tick_labels,
        rotation=0,
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=10,
        fontweight="bold"
    )

    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        length=6,
        width=1.8,
        direction="out",
        pad=7
    )

    ax.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        length=5,
        width=1.8,
        direction="out",
        pad=6
    )


    # --------------------------------------------------------
    # HEATMAP BORDER
    # --------------------------------------------------------

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.0)


    # --------------------------------------------------------
    # COLOUR-BAR FORMATTING
    # --------------------------------------------------------

    colorbar = heatmap.collections[0].colorbar

    colorbar.set_label(
        r"MHR target − baseline (beats min$^{-1}$)",
        fontsize=12,
        fontweight="bold",
        labelpad=13
    )

    colorbar.ax.tick_params(
        labelsize=10,
        width=1.5,
        length=5,
        direction="out"
    )

    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    colorbar.outline.set_visible(True)
    colorbar.outline.set_edgecolor("black")
    colorbar.outline.set_linewidth(1.5)


    # --------------------------------------------------------
    # DIRECTION LEGEND
    # --------------------------------------------------------

    direction_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=10,
            markerfacecolor="#009E73",
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=r"NAI $\rightarrow$ AI"
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=10,
            markerfacecolor="#E69F00",
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=r"AI $\rightarrow$ NAI"
        )
    ]


    legend.get_frame().set_linewidth(1.5)

    for text in legend.get_texts():
        text.set_fontweight("bold")

    legend.get_title().set_fontweight("bold")


    # --------------------------------------------------------
    # FIGURE NOTE
    # --------------------------------------------------------


    fig.subplots_adjust(
        left=0.12,
        right=0.94,
        bottom=0.17,
        top=0.82
    )


    # --------------------------------------------------------
    # SAVE PNG AND VECTOR PDF
    # --------------------------------------------------------

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    png_path = (
        output_path
        / "Figure_S5_switch_heterogeneity_nature.png"
    )

    pdf_path = (
        output_path
        / "Figure_S5_switch_heterogeneity_nature.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


print(f"PNG saved to: {png_path}")
print(f"PDF saved to: {pdf_path}")

# %% Notebook code cell 19
# Latency robustness grid. Leaving-direction latency should be interpreted only when its delta is coherent.
sens=[]
for smooth_s,fraction,min_delta in itertools.product([3,5,7],[.50,.63,.90],[.5,1.0,1.5]):
    for sid in subjects:
        for sw in switch_list(OFFSETS.get(sid,0.0)):
            tu,h=hr_trajectory(sid,sw["t"],smooth_s=smooth_s)
            est=estimate_latency(tu,h,sw["t"],fraction=fraction,min_delta=min_delta)
            sens.append({"smooth_s":smooth_s,"fraction":fraction,"min_delta":min_delta,
                         "subject":sid,"sw":sw["sw"],"direction":sw["dir"],**est})
latency_sensitivity=pd.DataFrame(sens)
latency_sensitivity.to_csv(Path(OUTPUT_DIR)/"Table_S5b_latency_sensitivity.csv",index=False)
display(latency_sensitivity.groupby(["direction","smooth_s","fraction","min_delta"])
        .agg(valid_latency=("latency","count"),median_latency=("latency","median"),
             median_delta=("delta","median")).reset_index())



latency_coverage = (latency.groupby("dir")
    .agg(participants=("subject", "nunique"), boundaries=("delta", "size"),
         valid_latency=("latency", "count"), median_latency=("latency", "median"),
         median_delta=("delta", "median")).reset_index())
latency_coverage.to_csv(Path(OUTPUT_DIR)/"Table_S5d_latency_coverage_all_participants.csv", index=False)
display(latency_coverage.round(3))

# %% Notebook code cell 21
rows = []
for sid, d in CACHE.items():
    timeline = build_timeline(OFFSETS.get(sid, 0.0))
    for _, clip_row in timeline.iterrows():
        first_start = float(clip_row["start"] + GUARD_S)
        last_start = float(clip_row["end"] - GUARD_S - WINDOW_S)
        if last_start < first_start:
            continue
        # Clip-anchored construction guarantees that no state window crosses a transition.
        for ws in np.arange(first_start, last_start + 1e-9, STEP_S):
            we = ws + WINDOW_S
            feat = feature_window(sid, ws, we, min_beats=MIN_BEATS_STATE)
            if feat is None:
                continue
            rows.append({
                "subject": sid, "start": ws, "end": we,
                "time_fraction": ws/VIDEO_TOTAL_S,
                "clip": int(clip_row["clip"]), "clip_desc": clip_row["desc"],
                "ai": int(clip_row["ai"]), "purity": 1.0, **feat
            })

state_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
assert set(FULL_FEATS).issubset(state_df.columns)
assert state_df["subject"].nunique() == len(subjects), "At least one participant has no valid state windows"
state_df["window_id"] = np.arange(len(state_df))
state_df.to_csv(Path(OUTPUT_DIR)/"state_features_all_original_features.csv", index=False)
print(state_df.groupby(["subject", "ai"]).size().unstack(fill_value=0))
print("windows:", len(state_df), "participants:", state_df["subject"].nunique(),
      "prevalence:", state_df["ai"].mean())

# %% Notebook code cell 23
def clip_subject_weights(frame):
    # Each subject-clip unit contributes equal total weight. With four AI and four NAI clips,
    # this already balances the two conditions; an extra class-frequency factor would double-correct.
    counts=frame.groupby(["subject","clip"])["ai"].transform("size").to_numpy()
    weights=1.0/counts
    return weights/np.mean(weights)

def base_pipeline(model):
    return Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),
                     ("scale",StandardScaler()),("clf",model)])

PRIMARY_MODEL=LogisticRegression(C=1.0,max_iter=3000,random_state=SEED)

def choose_threshold(y,p):
    # Youden/balanced-accuracy threshold chosen only from inner training predictions.
    y=np.asarray(y,int);p=np.asarray(p,float);keep=np.isfinite(p);y,p=y[keep],p[keep]
    fpr,tpr,thresholds=roc_curve(y,p)
    valid=np.isfinite(thresholds)
    if not np.any(valid):return 0.5
    idx=np.argmax((tpr-fpr)[valid])
    return float(thresholds[valid][idx])

def fit_probability_calibrator(y,p):
    # Fold-local calibration-in-the-large. Estimate an intercept on inner OOF predictions while
    # fixing the slope at 1, so class/clip weighting is corrected without reversing score ranking.
    y=np.asarray(y,int);p=np.asarray(p,float);keep=np.isfinite(p);y=y[keep]
    p=np.clip(p[keep],1e-6,1-1e-6);logit=np.log(p/(1-p));target=float(np.mean(y))
    lo,hi=-35.0,35.0
    for _ in range(80):
        mid=(lo+hi)/2;mean=float(np.mean(1/(1+np.exp(-np.clip(logit+mid,-35,35)))))
        if mean<target:lo=mid
        else:hi=mid
    return float((lo+hi)/2),1.0

def apply_probability_calibrator(p,intercept,slope):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6);logit=np.log(p/(1-p))
    z=np.clip(intercept+slope*logit,-35,35)
    return 1/(1+np.exp(-z))


def state_clip_probabilities(frame, probability):
    z = frame[["subject", "clip", "ai"]].copy()
    z["probability"] = np.asarray(probability, float)
    return (z.groupby(["subject", "clip", "ai"], as_index=False)
            .agg(probability=("probability", "median")))

def state_selection_auc(frame, probability):
    clip = state_clip_probabilities(frame, probability)
    aucs = [roc_auc_score(g["ai"], g["probability"])
            for _, g in clip.groupby("subject") if g["ai"].nunique() == 2]
    return float(np.mean(aucs))

def state_calibration_data(frame, probability):
    clip = state_clip_probabilities(frame, probability)
    return clip["ai"].to_numpy(int), clip["probability"].to_numpy(float)

def state_inner_oof(frame,features,model,n_splits=5):
    X=frame[features];y=frame["ai"].to_numpy(int);g=frame["subject"].to_numpy()
    oof=np.full(len(frame),np.nan)
    inner=GroupKFold(n_splits=min(n_splits,len(np.unique(g))))
    for tr,va in inner.split(X,y,g):
        pipe=base_pipeline(clone(model))
        pipe.fit(X.iloc[tr],y[tr],clf__sample_weight=clip_subject_weights(frame.iloc[tr]))
        oof[va]=pipe.predict_proba(X.iloc[va])[:,1]
    return oof

def fit_predict_loso(frame,features,model=PRIMARY_MODEL):
    X=frame[features]; y=frame["ai"].to_numpy(int); groups=frame["subject"].to_numpy()
    pred=np.full(len(frame),np.nan); predicted_class=np.full(len(frame),-1,int); chosen=[]
    for fold,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,groups)):
        inner_frame=frame.iloc[tr].reset_index(drop=True)
        inner_prob=state_inner_oof(inner_frame,features,model)
        cal_y,cal_p=state_calibration_data(inner_frame,inner_prob)
        cal_intercept,cal_slope=fit_probability_calibrator(cal_y,cal_p)
        inner_calibrated=apply_probability_calibrator(cal_p,cal_intercept,cal_slope)
        threshold=choose_threshold(cal_y,inner_calibrated)
        pipe=base_pipeline(clone(model))
        pipe.fit(X.iloc[tr],y[tr],clf__sample_weight=clip_subject_weights(frame.iloc[tr]))
        raw=pipe.predict_proba(X.iloc[te])[:,1]
        pred[te]=apply_probability_calibrator(raw,cal_intercept,cal_slope)
        predicted_class[te]=(pred[te]>=threshold).astype(int)
        chosen.append({"fold":fold,"test_subject":groups[te][0],"model":"fixed_logistic",
                       "inner_threshold":threshold,"inner_auc":state_selection_auc(inner_frame,inner_prob),
                       "calibration_intercept":cal_intercept,"calibration_slope":cal_slope})
    return pred,predicted_class,pd.DataFrame(chosen)

def candidate_models():
    models={
      "LogReg_C0.1":LogisticRegression(C=.1,max_iter=3000,random_state=SEED),
      "LogReg_C1":LogisticRegression(C=1,max_iter=3000,random_state=SEED),
      "SVM_C1":SVC(C=1,kernel="rbf",probability=True,random_state=SEED),
      "RF_depth4":RandomForestClassifier(n_estimators=500,max_depth=4,random_state=SEED,n_jobs=-1)
    }
    if HAVE_XGBOOST:
        models["XGB_depth2"]=XGBClassifier(n_estimators=300,max_depth=2,learning_rate=.05,
            subsample=.8,colsample_bytree=.8,eval_metric="logloss",random_state=SEED,n_jobs=-1)
    return models

def nested_loso(frame,features,label_col="ai",models=None):
    if models is None:
        models=candidate_models()
    X=frame[features]; y=frame[label_col].to_numpy(int)
    groups=frame["subject"].to_numpy(); pred=np.full(len(frame),np.nan)
    predicted_class=np.full(len(frame),-1,int); selection=[]
    for outer,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,groups)):
        tr_groups=groups[tr]; unique=np.unique(tr_groups); nsplit=min(5,len(unique))
        inner=GroupKFold(n_splits=nsplit); oof_by_model={name:np.full(len(tr),np.nan) for name in models}
        for itr,iva in inner.split(X.iloc[tr],y[tr],tr_groups):
            train_idx=tr[itr]; val_idx=tr[iva]
            for name,model in models.items():
                pipe=base_pipeline(clone(model))
                pipe.fit(X.iloc[train_idx],y[train_idx],
                         clf__sample_weight=clip_subject_weights(frame.iloc[train_idx]))
                oof_by_model[name][iva]=pipe.predict_proba(X.iloc[val_idx])[:,1]
        mean_scores={name:state_selection_auc(frame.iloc[tr],pv) for name,pv in oof_by_model.items()}
        selected=max(mean_scores,key=mean_scores.get)
        cal_y,cal_p=state_calibration_data(frame.iloc[tr],oof_by_model[selected])
        cal_intercept,cal_slope=fit_probability_calibrator(cal_y,cal_p)
        inner_calibrated=apply_probability_calibrator(cal_p,cal_intercept,cal_slope)
        threshold=choose_threshold(cal_y,inner_calibrated)
        final=base_pipeline(clone(models[selected]))
        final.fit(X.iloc[tr],y[tr],clf__sample_weight=clip_subject_weights(frame.iloc[tr]))
        raw=final.predict_proba(X.iloc[te])[:,1]
        pred[te]=apply_probability_calibrator(raw,cal_intercept,cal_slope)
        predicted_class[te]=(pred[te]>=threshold).astype(int)
        selection.append({"outer_fold":outer,"test_subject":groups[te][0],"selected":selected,
                          "inner_threshold":threshold,"calibration_intercept":cal_intercept,
                          "calibration_slope":cal_slope,
                          **{f"inner_{k}":v for k,v in mean_scores.items()}})
    return pred,predicted_class,pd.DataFrame(selection)

primary_prob,primary_class,primary_selection=fit_predict_loso(state_df,CORE_FEATS)
state_df["primary_probability"]=primary_prob
state_df["primary_class"]=primary_class
if RUN_SECONDARY_MODELS:
    secondary_prob,secondary_class,secondary_selection=nested_loso(state_df,FULL_FEATS)
    state_df["secondary_probability"]=secondary_prob
    state_df["secondary_class"]=secondary_class
else:
    secondary_selection=pd.DataFrame(); state_df["secondary_probability"]=np.nan
    state_df["secondary_class"]=-1


primary_threshold_map = primary_selection.set_index("test_subject")["inner_threshold"].to_dict()
state_df["primary_threshold"] = state_df["subject"].map(primary_threshold_map).astype(float)
if RUN_SECONDARY_MODELS:
    secondary_threshold_map = secondary_selection.set_index("test_subject")["inner_threshold"].to_dict()
    state_df["secondary_threshold"] = state_df["subject"].map(secondary_threshold_map).astype(float)
else:
    state_df["secondary_threshold"] = np.nan

# %% Notebook code cell 24
def expected_calibration_error(y, p, n_bins=10):
    y=np.asarray(y,int); p=np.asarray(p,float)
    bins=pd.qcut(pd.Series(p), q=min(n_bins,len(np.unique(p))), duplicates="drop")
    tmp=pd.DataFrame({"y":y,"p":p,"bin":bins})
    agg=tmp.groupby("bin",observed=True).agg(n=("y","size"),observed=("y","mean"),predicted=("p","mean"))
    return float(np.sum(agg["n"]/len(tmp)*np.abs(agg["observed"]-agg["predicted"])))

def calibration_intercept_slope(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    logit=np.log(p/(1-p)).reshape(-1,1)
    try:
        cal=LogisticRegression(C=1e6,max_iter=3000).fit(logit,y)
        return float(cal.intercept_[0]),float(cal.coef_[0,0])
    except Exception:
        return np.nan,np.nan

def prediction_metrics(y,p,predicted_class=None,threshold=.5):
    keep=np.isfinite(p); y=np.asarray(y)[keep]; p=np.asarray(p)[keep]
    pred=(p>=threshold).astype(int) if predicted_class is None else np.asarray(predicted_class)[keep].astype(int)
    brier=brier_score_loss(y,p); null_brier=brier_score_loss(y,np.full(len(y),np.mean(y)))
    cal_intercept,cal_slope=calibration_intercept_slope(y,p)
    return {"n":len(y),"prevalence":np.mean(y),"roc_auc":roc_auc_score(y,p),
            "pr_auc":average_precision_score(y,p),"balanced_accuracy":balanced_accuracy_score(y,pred),
            "macro_f1":f1_score(y,pred,average="macro"),"brier":brier,
            "brier_skill":1-brier/null_brier if null_brier>0 else np.nan,
            "calibration_intercept":cal_intercept,"calibration_slope":cal_slope,
            "ece":expected_calibration_error(y,p)}

def make_clip_predictions(frame, probability_col, threshold_col):
    return (frame.groupby(["subject","clip","ai"],as_index=False)
        .agg(probability=(probability_col,"median"),threshold=(threshold_col,"median"),
             n_windows=("window_id","size"))
        .assign(predicted_class=lambda x:(x["probability"]>=x["threshold"]).astype(int)))

def cluster_prediction_cis(frame,label_col,prob_col,class_col,n=N_BOOT,seed=SEED):
    rng=np.random.default_rng(seed); subs=frame["subject"].unique()
    metrics={k:[] for k in ["roc_auc","pr_auc","balanced_accuracy","macro_f1","brier"]}
    for _ in range(n):
        picked=rng.choice(subs,len(subs),replace=True); yy=[];pp=[];cc=[]
        for s in picked:
            g=frame[frame["subject"]==s]
            yy.extend(g[label_col]);pp.extend(g[prob_col]);cc.extend(g[class_col])
        if len(np.unique(yy))!=2: continue
        metrics["roc_auc"].append(roc_auc_score(yy,pp))
        metrics["pr_auc"].append(average_precision_score(yy,pp))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(yy,cc))
        metrics["macro_f1"].append(f1_score(yy,cc,average="macro"))
        metrics["brier"].append(brier_score_loss(yy,pp))
    out={}
    for metric,vals in metrics.items():
        out[f"{metric}_ci_lo"],out[f"{metric}_ci_hi"]=np.percentile(vals,[2.5,97.5])
    return out

def subject_auc_summary(frame,prob_col,label_col="ai",n=N_BOOT,seed=SEED):
    aucs=np.asarray([roc_auc_score(g[label_col],g[prob_col]) for _,g in frame.groupby("subject")
                     if g[label_col].nunique()==2],float)
    rng=np.random.default_rng(seed)
    boot=np.array([np.mean(rng.choice(aucs,len(aucs),replace=True)) for _ in range(n)])
    return {"mean_subject_auc":np.mean(aucs),"median_subject_auc":np.median(aucs),
            "subject_auc_min":np.min(aucs),"subject_auc_max":np.max(aucs),
            "mean_subject_auc_ci_lo":np.percentile(boot,2.5),
            "mean_subject_auc_ci_hi":np.percentile(boot,97.5)}

primary_clip=make_clip_predictions(state_df,"primary_probability","primary_threshold")
secondary_clip=(make_clip_predictions(state_df,"secondary_probability","secondary_threshold")
                if RUN_SECONDARY_MODELS else pd.DataFrame())

model_rows=[]
for name,clip in [("Primary fixed logistic — participant×clip",primary_clip),
                  ("Secondary nested model — participant×clip",secondary_clip)]:
    if len(clip):
        model_rows.append({"analysis":name,"evaluation_unit":"participant_clip",
            **prediction_metrics(clip["ai"],clip["probability"],clip["predicted_class"]),
            **subject_auc_summary(clip,"probability"),
            **cluster_prediction_cis(clip,"ai","probability","predicted_class")})

# Window estimates are retained only as a sensitivity analysis.
model_rows.append({"analysis":"Primary fixed logistic — window sensitivity",
    "evaluation_unit":"overlapping_window",
    **prediction_metrics(state_df["ai"],state_df["primary_probability"],state_df["primary_class"]),
    **subject_auc_summary(state_df,"primary_probability"),
    **cluster_prediction_cis(state_df,"ai","primary_probability","primary_class")})

model_results=pd.DataFrame(model_rows)
model_results.to_csv(Path(OUTPUT_DIR)/"Table_3_state_prediction_results.csv",index=False)
primary_clip.to_csv(Path(OUTPUT_DIR)/"Table_S6_clip_level_outer_predictions.csv",index=False)
state_df[["subject","start","clip","ai","primary_probability","primary_threshold","primary_class",
          "secondary_probability","secondary_threshold","secondary_class"]].to_csv(
    Path(OUTPUT_DIR)/"Table_S6_outer_fold_window_predictions.csv",index=False)
primary_selection.to_csv(Path(OUTPUT_DIR)/"Table_S6b_primary_thresholds.csv",index=False)
secondary_selection.to_csv(Path(OUTPUT_DIR)/"Table_S7_nested_model_selection.csv",index=False)
display(model_results.round(3))

# ============================================================
# NATURE-STYLE PRIMARY STATE-MODEL FIGURE
# Run after primary_clip and model_results have been created
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)


# ------------------------------------------------------------
# COLOURS
# Distinct from the teal and orange used in Figure 2
# ------------------------------------------------------------

ROC_COLOR = "#6A3D9A"          # deep purple
PR_COLOR = "#CC79A7"           # reddish purple/magenta
CALIBRATION_COLOR = "#8E3B8F"  # plum
BAR_ABOVE_COLOR = "#1FB6B2"    # purple
BAR_BELOW_COLOR = "#D45087"    # magenta
REFERENCE_COLOR = "#303030"    # dark grey/black


# ------------------------------------------------------------
# FONT AND FIGURE STYLE
# ------------------------------------------------------------

nature_rc = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 22,
    "font.weight": "bold",

    "axes.labelsize": 22,
    "axes.labelweight": "bold",
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.linewidth": 2.8,

    "xtick.labelsize": 22,
    "ytick.labelsize": 22,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    # Preserve editable text in vector output
    "pdf.fonttype": 42,
    "ps.fonttype": 42
}


# ------------------------------------------------------------
# DATA AND SUMMARY METRICS
# ------------------------------------------------------------

y = primary_clip["ai"].to_numpy(dtype=int)
p = primary_clip["probability"].to_numpy(dtype=float)

roc_auc = roc_auc_score(y, p)
average_precision = average_precision_score(y, p)

fpr, tpr, _ = roc_curve(y, p)
precision, recall, _ = precision_recall_curve(y, p)

calibration_intercept, calibration_slope = (
    calibration_intercept_slope(y, p)
)

ece_value = expected_calibration_error(y, p)


# Five quantile-based calibration groups
number_of_calibration_bins = min(
    5,
    len(np.unique(p))
)

calibration_bins = pd.qcut(
    p,
    q=number_of_calibration_bins,
    duplicates="drop"
)

calibration_data = (
    pd.DataFrame({
        "observed": y,
        "predicted": p,
        "bin": calibration_bins
    })
    .groupby("bin", observed=True)
    .agg(
        observed_proportion=("observed", "mean"),
        mean_predicted_probability=("predicted", "mean"),
        n=("observed", "size")
    )
    .reset_index(drop=True)
)


# Participant-specific clip-level AUC
participant_auc = pd.DataFrame(
    [
        {
            "subject": subject,
            "auc": roc_auc_score(
                participant_data["ai"],
                participant_data["probability"]
            )
        }
        for subject, participant_data
        in primary_clip.groupby("subject")
        if participant_data["ai"].nunique() == 2
    ]
).sort_values(
    "auc",
    ascending=True
).reset_index(drop=True)


# Use different bar colours for values above and below chance
participant_auc["color"] = np.where(
    participant_auc["auc"] >= 0.5,
    BAR_ABOVE_COLOR,
    BAR_BELOW_COLOR
)


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def format_panel(ax):
    """Apply common Nature-style formatting to a subplot."""

    ax.grid(False)

    # Complete black box around the subplot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.8)

    ax.tick_params(
        axis="both",
        which="major",
        width=2.8,
        length=8,
        direction="out",
        pad=8
    )

    # Bold tick labels
    for tick_label in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        tick_label.set_fontweight("bold")


def format_legend(legend):
    """Format an internal boxed legend."""

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(2.0)
    legend.get_frame().set_alpha(1.0)


# ------------------------------------------------------------
# CREATE 2 × 2 FIGURE
# ------------------------------------------------------------

with plt.rc_context(nature_rc):

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(18, 15),
        constrained_layout=False
    )

    axes = axes.ravel()

    # ========================================================
    # PANEL A: ROC CURVE
    # ========================================================

    ax = axes[0]

    ax.plot(
        fpr,
        tpr,
        color=ROC_COLOR,
        linewidth=4.0,
        zorder=3
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color=REFERENCE_COLOR,
        linewidth=2.5,
        linestyle="--",
        zorder=2
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.set_xlabel(
        "False-positive rate",
        fontsize=22,
        fontweight="bold"
    )

    ax.set_ylabel(
        "True-positive rate",
        fontsize=22,
        fontweight="bold"
    )

    ax.set_title(
        f"ROC curve (AUC = {roc_auc:.3f})",
        fontsize=22,
        fontweight="bold",
        pad=16
    )

    roc_legend_handles = [
        Line2D(
            [0], [0],
            color=ROC_COLOR,
            linewidth=4,
            label="Primary ECG model"
        ),
        Line2D(
            [0], [0],
            color=REFERENCE_COLOR,
            linewidth=2.5,
            linestyle="--",
            label="Chance discrimination"
        )
    ]

    roc_legend = ax.legend(
        handles=roc_legend_handles,
        loc="lower right",
        frameon=True,
        fancybox=False,
        prop={
            "family": "sans-serif",
            "size": 17,
            "weight": "bold"
        },
        borderpad=0.55,
        labelspacing=0.35,
        handlelength=1.8
    )

    format_legend(roc_legend)
    format_panel(ax)


    # ========================================================
    # PANEL B: PRECISION–RECALL CURVE
    # ========================================================

    ax = axes[1]

    ax.plot(
        recall,
        precision,
        color=PR_COLOR,
        linewidth=4.0,
        zorder=3
    )

    ax.axhline(
        y=np.mean(y),
        color=REFERENCE_COLOR,
        linewidth=2.5,
        linestyle="--",
        zorder=2
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.set_xlabel(
        "Recall",
        fontsize=22,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Precision",
        fontsize=22,
        fontweight="bold"
    )

    ax.set_title(
        f"Precision–recall curve (AP = {average_precision:.3f})",
        fontsize=22,
        fontweight="bold",
        pad=16
    )

    pr_legend_handles = [
        Line2D(
            [0], [0],
            color=PR_COLOR,
            linewidth=4,
            label="Primary ECG model"
        ),
        Line2D(
            [0], [0],
            color=REFERENCE_COLOR,
            linewidth=2.5,
            linestyle="--",
            label=f"Prevalence = {np.mean(y):.2f}"
        )
    ]

    pr_legend = ax.legend(
        handles=pr_legend_handles,
        loc="upper right",
        frameon=True,
        fancybox=False,
        prop={
            "family": "sans-serif",
            "size": 17,
            "weight": "bold"
        },
        borderpad=0.55,
        labelspacing=0.35,
        handlelength=1.8
    )

    format_legend(pr_legend)
    format_panel(ax)


    # ========================================================
    # PANEL C: CALIBRATION
    # ========================================================

    ax = axes[2]

    # Perfect calibration reference
    ax.plot(
        [0, 1],
        [0, 1],
        color=REFERENCE_COLOR,
        linewidth=2.5,
        linestyle="--",
        zorder=2
    )

    # Observed calibration curve
    ax.plot(
        calibration_data["mean_predicted_probability"],
        calibration_data["observed_proportion"],
        color=CALIBRATION_COLOR,
        linewidth=4.0,
        marker="o",
        markersize=12,
        markerfacecolor=CALIBRATION_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.7,
        zorder=4
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.set_xlabel(
        "Mean predicted probability",
        fontsize=22,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Observed proportion",
        fontsize=22,
        fontweight="bold"
    )

    ax.set_title(
        "Clip-level calibration",
        fontsize=22,
        fontweight="bold",
        pad=16
    )

    calibration_legend_handles = [
        Line2D(
            [0], [0],
            color=CALIBRATION_COLOR,
            linewidth=4,
            marker="o",
            markersize=10,
            label="Observed calibration"
        ),
        Line2D(
            [0], [0],
            color=REFERENCE_COLOR,
            linewidth=2.5,
            linestyle="--",
            label="Perfect calibration"
        )
    ]

    calibration_legend = ax.legend(
        handles=calibration_legend_handles,
        loc="upper left",
        frameon=True,
        fancybox=False,
        prop={
            "family": "sans-serif",
            "size": 17,
            "weight": "bold"
        },
        borderpad=0.55,
        labelspacing=0.35,
        handlelength=1.8
    )

    format_legend(calibration_legend)

    # Calibration statistics box
    calibration_text = (
        f"Intercept = {calibration_intercept:.3f}\n"
        f"Slope = {calibration_slope:.3f}\n"
        f"ECE = {ece_value:.3f}"
    )

    ax.text(
        0.97,
        0.05,
        calibration_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        bbox={
            "boxstyle": "square,pad=0.45",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 2.0,
            "alpha": 1.0
        }
    )

    format_panel(ax)


    # ========================================================
    # PANEL D: PARTICIPANT-SPECIFIC AUC
    # ========================================================

    ax = axes[3]

    y_positions = np.arange(len(participant_auc))

    ax.barh(
        y_positions,
        participant_auc["auc"],
        color=participant_auc["color"],
        edgecolor="black",
        linewidth=1.2,
        height=0.72,
        zorder=3
    )

    ax.axvline(
        x=0.5,
        color=REFERENCE_COLOR,
        linewidth=2.5,
        linestyle="--",
        zorder=4
    )

    ax.set_yticks(y_positions)

    ax.set_yticklabels(
        participant_auc["subject"],
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))

    # Create internal space above the bars for the legend
    ax.set_ylim(
        -0.8,
        len(participant_auc) + 2.8
    )

    ax.set_xlabel(
        "ROC AUC",
        fontsize=22,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Participant",
        fontsize=22,
        fontweight="bold"
    )

    ax.set_title(
        "Participant-specific clip AUC",
        fontsize=22,
        fontweight="bold",
        pad=16
    )

    participant_legend_handles = [
        Patch(
            facecolor=BAR_ABOVE_COLOR,
            edgecolor="black",
            label=r"AUC $\geq$ 0.50"
        ),
        Patch(
            facecolor=BAR_BELOW_COLOR,
            edgecolor="black",
            label="AUC < 0.50"
        ),
        Line2D(
            [0], [0],
            color=REFERENCE_COLOR,
            linewidth=2.5,
            linestyle="--",
            label="Chance AUC"
        )
    ]

    participant_legend = ax.legend(
        handles=participant_legend_handles,
        loc="upper left",
        ncol=3,
        frameon=True,
        fancybox=False,
        prop={
            "family": "sans-serif",
            "size": 14,
            "weight": "bold"
        },
        borderpad=0.50,
        labelspacing=0.30,
        handletextpad=0.40,
        columnspacing=0.8,
        handlelength=1.4
    )

    format_legend(participant_legend)
    format_panel(ax)


    # ========================================================
    # PANEL LABELS: a, b, c, d
    # ========================================================

    for panel_label, ax in zip(
        ["a", "b", "c", "d"],
        axes
    ):
        ax.text(
            -0.16,
            1.10,
            panel_label,
            transform=ax.transAxes,
            fontsize=26,
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False
        )


    # --------------------------------------------------------
    # FIGURE SPACING
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.08,
        top=0.94,
        wspace=0.30,
        hspace=0.36
    )


    # --------------------------------------------------------
    # SAVE FIGURE
    # --------------------------------------------------------

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    png_path = (
        Path(OUTPUT_DIR)
        / "Figure_3_primary_state_model.png"
    )

    pdf_path = (
        Path(OUTPUT_DIR)
        / "Figure_3_primary_state_model_nature.pdf"
    )

    # High-resolution PNG
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    # Vector PDF
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


print(f"PNG saved to: {png_path}")
print(f"Vector PDF saved to: {pdf_path}")

# %% Notebook code cell 26
def loso_control_numeric(frame,features,model=PRIMARY_MODEL):
    p,c,folds=fit_predict_loso(frame,features,model=model)
    threshold_map=folds.set_index("test_subject")["inner_threshold"].to_dict()
    return p,frame["subject"].map(threshold_map).to_numpy(float)

def loso_clip_only(frame):
    y=frame["ai"].to_numpy(int); groups=frame["subject"].to_numpy()
    p=np.full(len(frame),np.nan); thresholds=np.full(len(frame),np.nan)
    def make_clip_model():
        pre=ColumnTransformer([("clip",OneHotEncoder(handle_unknown="ignore"),["clip"])])
        return Pipeline([("prep",pre),("clf",LogisticRegression(max_iter=2000))])
    for tr,te in LeaveOneGroupOut().split(frame,y,groups):
        inner_frame=frame.iloc[tr].reset_index(drop=True); iy=inner_frame["ai"].to_numpy(int)
        ig=inner_frame["subject"].to_numpy(); oof=np.full(len(inner_frame),np.nan)
        for a,b in GroupKFold(n_splits=min(5,len(np.unique(ig)))).split(inner_frame,iy,ig):
            m=make_clip_model();m.fit(inner_frame.iloc[a],iy[a]);oof[b]=m.predict_proba(inner_frame.iloc[b])[:,1]
        cal_y,cal_p=state_calibration_data(inner_frame,oof)
        intercept,slope=fit_probability_calibrator(cal_y,cal_p)
        threshold=choose_threshold(cal_y,apply_probability_calibrator(cal_p,intercept,slope))
        model=make_clip_model();model.fit(frame.iloc[tr],y[tr])
        p[te]=apply_probability_calibrator(model.predict_proba(frame.iloc[te])[:,1],intercept,slope)
        thresholds[te]=threshold
    return p,thresholds

controls={
    "Primary ECG":(state_df["primary_probability"].to_numpy(),state_df["primary_threshold"].to_numpy()),
    "MHR only":loso_control_numeric(state_df,["MHR"]),
    "Time only":loso_control_numeric(state_df,["time_fraction"],
        model=RandomForestClassifier(n_estimators=500,max_depth=4,random_state=SEED,n_jobs=-1)),
    "Window RR quality only":loso_control_numeric(state_df,["window_rr_correction_fraction"]),
    "Clip identity only":loso_clip_only(state_df)
}
control_rows=[]
for name,(p,thresholds) in controls.items():
    temp=state_df.copy();temp["_p"]=p;temp["_t"]=thresholds
    clip=(temp.groupby(["subject","clip","ai"],as_index=False)
          .agg(probability=("_p","median"),threshold=("_t","median"))
          .assign(predicted_class=lambda x:(x["probability"]>=x["threshold"]).astype(int)))
    control_rows.append({"control":name,"evaluation_unit":"participant_clip",
        **prediction_metrics(clip["ai"],clip["probability"],clip["predicted_class"]),
        **cluster_prediction_cis(clip,"ai","probability","predicted_class")})
control_results=pd.DataFrame(control_rows)

# Strict double holdout: the test participant and both held-out clip identities are absent from training.
pair_window_predictions=[]
ai_clips=[c for c,a,*_ in CLIPS if a==1]; nai_clips=[c for c,a,*_ in CLIPS if a==0]
for test_subject in state_df["subject"].unique():
    for ai_clip,nai_clip in itertools.product(ai_clips,nai_clips):
        held=[ai_clip,nai_clip]
        train=state_df[(state_df["subject"]!=test_subject)&(~state_df["clip"].isin(held))].reset_index(drop=True)
        test=state_df[(state_df["subject"]==test_subject)&(state_df["clip"].isin(held))]
        if train["ai"].nunique()<2 or test["ai"].nunique()<2: continue
        inner=state_inner_oof(train,CORE_FEATS,PRIMARY_MODEL)
        cal_y,cal_p=state_calibration_data(train,inner)
        intercept,slope=fit_probability_calibrator(cal_y,cal_p)
        threshold=choose_threshold(cal_y,apply_probability_calibrator(cal_p,intercept,slope))
        model=base_pipeline(clone(PRIMARY_MODEL))
        model.fit(train[CORE_FEATS],train["ai"],clf__sample_weight=clip_subject_weights(train))
        pp=apply_probability_calibrator(model.predict_proba(test[CORE_FEATS])[:,1],intercept,slope)
        for (_,r),value in zip(test.iterrows(),pp):
            pair_window_predictions.append({"subject":test_subject,"ai_clip":ai_clip,"nai_clip":nai_clip,
                "clip":r["clip"],"true":r["ai"],"probability":value,"threshold":threshold})

pair_window_predictions=pd.DataFrame(pair_window_predictions)
pair_clip_predictions=(pair_window_predictions.groupby(
    ["subject","ai_clip","nai_clip","clip","true"],as_index=False)
    .agg(probability=("probability","median"),threshold=("threshold","median"),n_windows=("probability","size"))
    .assign(predicted_class=lambda x:(x["probability"]>=x["threshold"]).astype(int)))
pair_rows=[]
for (sid,ai_clip,nai_clip),g in pair_clip_predictions.groupby(["subject","ai_clip","nai_clip"]):
    if g["true"].nunique()==2:
        pair_rows.append({"subject":sid,"ai_clip":ai_clip,"nai_clip":nai_clip,
                          **prediction_metrics(g["true"],g["probability"],g["predicted_class"])})
pair_subject_results=pd.DataFrame(pair_rows)
subject_pair_means=pair_subject_results.groupby("subject").mean(numeric_only=True)
rng=np.random.default_rng(SEED); boot=[]
for _ in range(N_BOOT):
    sampled=rng.choice(subject_pair_means.index,len(subject_pair_means),replace=True)
    boot.append(subject_pair_means.loc[sampled,["roc_auc","balanced_accuracy","brier"]].mean().to_dict())
boot=pd.DataFrame(boot)
macro={k:pair_subject_results[k].mean() for k in ["roc_auc","pr_auc","balanced_accuracy","macro_f1","brier"]}
control_results.loc[len(control_results)]={"control":"Unseen participant + unseen clips",
    "evaluation_unit":"participant_clip_pair","n":len(pair_clip_predictions),
    "prevalence":pair_clip_predictions["true"].mean(),**macro,
    "roc_auc_ci_lo":boot["roc_auc"].quantile(.025),"roc_auc_ci_hi":boot["roc_auc"].quantile(.975),
    "balanced_accuracy_ci_lo":boot["balanced_accuracy"].quantile(.025),
    "balanced_accuracy_ci_hi":boot["balanced_accuracy"].quantile(.975),
    "brier_ci_lo":boot["brier"].quantile(.025),"brier_ci_hi":boot["brier"].quantile(.975)}

control_results.to_csv(Path(OUTPUT_DIR)/"Table_S8_confounding_and_generalisation_controls.csv",index=False)
pair_window_predictions.to_csv(Path(OUTPUT_DIR)/"Table_S8b_unseen_subject_stimulus_window_predictions.csv",index=False)
pair_clip_predictions.to_csv(Path(OUTPUT_DIR)/"Table_S8c_unseen_subject_stimulus_clip_predictions.csv",index=False)
pair_subject_results.to_csv(Path(OUTPUT_DIR)/"Table_S8d_unseen_clip_pair_metrics.csv",index=False)
display(control_results.round(3))

fig,ax=plt.subplots(figsize=(8,4.5)); z=control_results.sort_values("roc_auc")
ax.barh(z["control"],z["roc_auc"],color="#4472c4"); ax.axvline(.5,color="black",ls="--")
ax.set(xlim=(0,1),xlabel="Held-out ROC AUC",title="ECG model versus protocol controls")
plt.tight_layout(); plt.savefig(Path(OUTPUT_DIR)/"Figure_4_confounding_controls.png",dpi=300); plt.show()

# %% Notebook code cell 27
# ============================================================
# NATURE-STYLE CONFOUNDING AND GENERALISATION FIGURE
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ------------------------------------------------------------
# LOAD RESULTS IF THEY ARE NOT PRESENT IN MEMORY
# ------------------------------------------------------------

control_results_file = (
    Path(OUTPUT_DIR)
    / "Table_S8_confounding_and_generalisation_controls.csv"
)

if "control_results" not in globals():

    if control_results_file.exists():

        control_results = pd.read_csv(
            control_results_file
        )

        print(
            "Loaded control results from:",
            control_results_file
        )

    else:

        raise RuntimeError(
            "control_results is not available and the saved "
            "Table_S8_confounding_and_generalisation_controls.csv "
            "file could not be found. Run the control-analysis "
            "cell before generating this figure."
        )


# ------------------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = {
    "control",
    "roc_auc",
    "roc_auc_ci_lo",
    "roc_auc_ci_hi"
}

missing_columns = (
    required_columns
    - set(control_results.columns)
)

if missing_columns:

    raise RuntimeError(
        "control_results is missing the following columns: "
        f"{sorted(missing_columns)}"
    )


plot_data = (
    control_results
    .dropna(subset=["control", "roc_auc"])
    .copy()
)

plot_data["roc_auc"] = pd.to_numeric(
    plot_data["roc_auc"],
    errors="coerce"
)

plot_data["roc_auc_ci_lo"] = pd.to_numeric(
    plot_data["roc_auc_ci_lo"],
    errors="coerce"
)

plot_data["roc_auc_ci_hi"] = pd.to_numeric(
    plot_data["roc_auc_ci_hi"],
    errors="coerce"
)

plot_data = (
    plot_data
    .dropna(subset=["roc_auc"])
    .sort_values("roc_auc", ascending=True)
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# DISPLAY LABELS
# ------------------------------------------------------------

DISPLAY_LABELS = {
    "Primary ECG": "Primary ECG",
    "MHR only": "MHR only",
    "Time only": "Protocol time only",
    "Window RR quality only": "RR quality only",
    "Clip identity only": "Clip identity only",
    "Unseen participant + unseen clips":
        "Unseen participant +\nunseen clips"
}

plot_data["display_label"] = (
    plot_data["control"]
    .map(DISPLAY_LABELS)
    .fillna(plot_data["control"])
)


# ------------------------------------------------------------
# COLOUR PALETTE
# Distinct from the previous teal–orange and purple–magenta
# figures
# ------------------------------------------------------------

CONTROL_COLORS = {
    "Primary ECG": "#8B1E3F",            # burgundy
    "MHR only": "#B07C91",               # muted rose
    "Time only": "#555555",               # charcoal
    "Window RR quality only": "#BDBDBD", # light grey
    "Clip identity only": "#555555",      # charcoal
    "Unseen participant + unseen clips":
        "#A47C00"                         # dark gold
}

plot_data["color"] = (
    plot_data["control"]
    .map(CONTROL_COLORS)
    .fillna("#8C8C8C")
)


# ------------------------------------------------------------
# CALCULATE ASYMMETRIC CONFIDENCE-INTERVAL ERRORS
# ------------------------------------------------------------

# If a confidence limit is unavailable, display no error bar
plot_data["ci_lo_plot"] = (
    plot_data["roc_auc_ci_lo"]
    .fillna(plot_data["roc_auc"])
)

plot_data["ci_hi_plot"] = (
    plot_data["roc_auc_ci_hi"]
    .fillna(plot_data["roc_auc"])
)

lower_error = np.maximum(
    0,
    plot_data["roc_auc"]
    - plot_data["ci_lo_plot"]
)

upper_error = np.maximum(
    0,
    plot_data["ci_hi_plot"]
    - plot_data["roc_auc"]
)


# ------------------------------------------------------------
# NATURE-STYLE SETTINGS
# ------------------------------------------------------------

nature_rc = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "DejaVu Sans"
    ],
    "font.size": 22,
    "font.weight": "bold",

    "axes.labelsize": 22,
    "axes.labelweight": "bold",
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.linewidth": 2.8,

    "xtick.labelsize": 22,
    "ytick.labelsize": 19,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    # Preserve editable fonts in vector output
    "pdf.fonttype": 42,
    "ps.fonttype": 42
}


# ------------------------------------------------------------
# CREATE FIGURE
# ------------------------------------------------------------

with plt.rc_context(nature_rc):

    fig, ax = plt.subplots(
        figsize=(16, 10),
        constrained_layout=False
    )

    y_positions = np.arange(
        len(plot_data)
    )


    # --------------------------------------------------------
    # HORIZONTAL BARS
    # --------------------------------------------------------

    ax.barh(
        y_positions,
        plot_data["roc_auc"],
        height=0.66,
        color=plot_data["color"],
        edgecolor="black",
        linewidth=1.7,
        zorder=2
    )


    # --------------------------------------------------------
    # 95% CONFIDENCE INTERVALS
    # --------------------------------------------------------

    ax.errorbar(
        x=plot_data["roc_auc"],
        y=y_positions,
        xerr=np.vstack([
            lower_error,
            upper_error
        ]),
        fmt="none",
        ecolor="black",
        elinewidth=3.5,
        capsize=7,
        capthick=3.5,
        zorder=4
    )


    # --------------------------------------------------------
    # CHANCE DISCRIMINATION
    # --------------------------------------------------------

    ax.axvline(
        x=0.5,
        color="black",
        linestyle="--",
        linewidth=3.0,
        zorder=3
    )


    # --------------------------------------------------------
    # NUMERICAL AUC LABELS
    # --------------------------------------------------------
    # --------------------------------------------------------
    # NUMERICAL AUC LABELS
    # Place labels beyond the right confidence-interval cap
    # --------------------------------------------------------

    label_offset = 0.025

    for y_position, auc_value, ci_high in zip(
        y_positions,
        plot_data["roc_auc"],
        plot_data["ci_hi_plot"]
    ):

        # Position label after the upper confidence-limit cap
        text_x = ci_high + label_offset

        ax.text(
            text_x,
            y_position,
            f"{auc_value:.3f}",
            ha="left",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="black",
            zorder=6,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
                "pad": 0.5
            }
        )


    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        1.10
    )

    ax.set_xticks(
        np.arange(0, 1.01, 0.2)
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        plot_data["display_label"],
        fontsize=19,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Held-out ROC AUC",
        fontsize=22,
        fontweight="bold",
        labelpad=14
    )

    ax.set_ylabel(
        "",
        fontsize=22,
        fontweight="bold"
    )

    ax.set_title(
        "State-model performance and protocol controls",
        fontsize=24,
        fontweight="bold",
        pad=18
    )


    # --------------------------------------------------------
    # INTERNAL SPACE FOR LEGEND
    # --------------------------------------------------------

    # Add blank space above the uppermost bar
    ax.set_ylim(
        -0.8,
        len(plot_data) + 2.1
    )


    # --------------------------------------------------------
    # INTERNAL BOXED LEGEND
    # --------------------------------------------------------

    legend_handles = [
        Patch(
            facecolor=CONTROL_COLORS["Primary ECG"],
            edgecolor="black",
            label="Primary ECG model"
        ),
        Patch(
            facecolor=CONTROL_COLORS["MHR only"],
            edgecolor="black",
            label="Reduced physiology"
        ),
        Patch(
            facecolor=CONTROL_COLORS["Time only"],
            edgecolor="black",
            label="Protocol diagnostic"
        ),
        Patch(
            facecolor=CONTROL_COLORS[
                "Window RR quality only"
            ],
            edgecolor="black",
            label="Quality-control model"
        ),
        Patch(
            facecolor=CONTROL_COLORS[
                "Unseen participant + unseen clips"
            ],
            edgecolor="black",
            label="Strict double holdout"
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            linewidth=3.0,
            label="Chance AUC = 0.50"
        )
    ]

    figure_legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        prop={
            "family": "sans-serif",
            "size": 15,
            "weight": "bold"
        },
        borderpad=0.65,
        labelspacing=0.40,
        handletextpad=0.45,
        columnspacing=1.0,
        handlelength=1.5
    )

    figure_legend.get_frame().set_linewidth(
        2.2
    )

    figure_legend.get_frame().set_edgecolor(
        "black"
    )


    # --------------------------------------------------------
    # REMOVE GRIDLINES AND THICKEN AXIS BOX
    # --------------------------------------------------------

    ax.grid(False)

    for spine in ax.spines.values():

        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.8)


    # --------------------------------------------------------
    # TICK FORMATTING
    # --------------------------------------------------------

    ax.tick_params(
        axis="both",
        which="major",
        width=2.8,
        length=8,
        direction="out",
        pad=9
    )

    for tick_label in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
    ):

        tick_label.set_fontweight(
            "bold"
        )


    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.28,
        right=0.97,
        bottom=0.14,
        top=0.90
    )


    # --------------------------------------------------------
    # SAVE FIGURE
    # --------------------------------------------------------

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    png_path = (
        Path(OUTPUT_DIR)
        / "Figure_4_confounding_controls.png"
    )

    pdf_path = (
        Path(OUTPUT_DIR)
        / "Figure_4_confounding_controls_nature.pdf"
    )

    # High-resolution PNG
    fig.savefig(
        png_path,
        dpi=900,
        bbox_inches="tight",
        facecolor="white"
    )

    # Vector PDF
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


print(f"PNG saved to: {png_path}")
print(f"Vector PDF saved to: {pdf_path}")

# %% Notebook code cell 28
# Exact clip-level randomisation: all 4-of-8 label assignments (70 possibilities).
def primary_auc_for_clip_assignment(frame,positive_clips):
    d=frame.copy();d["ai"]=d["clip"].isin(set(positive_clips)).astype(int)
    X=d[CORE_FEATS];y=d["ai"].to_numpy(int);groups=d["subject"].to_numpy();p=np.full(len(d),np.nan)
    for tr,te in LeaveOneGroupOut().split(X,y,groups):
        model=base_pipeline(clone(PRIMARY_MODEL))
        model.fit(X.iloc[tr],y[tr],clf__sample_weight=clip_subject_weights(d.iloc[tr]))
        p[te]=model.predict_proba(X.iloc[te])[:,1]
    return state_selection_auc(d,p)

observed_auc=state_selection_auc(state_df,state_df["primary_probability"])
assignment_rows=[]
for positive in itertools.combinations(range(1,9),4):
    assignment_rows.append({"positive_clips":"-".join(map(str,positive)),
                            "mean_subject_clip_auc":primary_auc_for_clip_assignment(state_df,positive)})
clip_null=pd.DataFrame(assignment_rows)
p_clip=float(np.mean(clip_null["mean_subject_clip_auc"]>=observed_auc-1e-12))
print(f"Observed mean participant clip-level AUC={observed_auc:.3f}; exact clip-label p={p_clip:.4f}")
clip_null.to_csv(Path(OUTPUT_DIR)/"Table_S8e_clip_label_randomisation.csv",index=False)

# %% Notebook code cell 30
def build_transition_dataset(direction,window_s=60.0):
    half=window_s/2;rows=[]
    for sid,d in CACHE.items():
        tl=build_timeline(OFFSETS.get(sid,0));subject_switches=switch_list(OFFSETS.get(sid,0))
        bounds=np.array([s["t"] for s in subject_switches]);switch_ids=np.array([s["sw"] for s in subject_switches])
        for sw in subject_switches:
            if sw["dir"]!=direction:continue
            ws,we=sw["t"]-half,sw["t"]+half
            feat=feature_window(sid,ws,we,include_transition=True,min_beats=MIN_BEATS_STATE)
            if feat:rows.append({"subject":sid,"start":ws,"center":sw["t"],"clip":sw["to"],
                "label":1,"time_fraction":sw["t"]/VIDEO_TOTAL_S,"nearest_switch":sw["sw"],
                "distance_to_boundary":0.0,**feat})
        for _,c in tl.iterrows():
            for center in np.arange(c["start"]+half,c["end"]-half+1e-9,TRANSITION_NEG_STEP_S):
                if len(bounds) and np.min(abs(center-bounds))<half+10:continue
                feat=feature_window(sid,center-half,center+half,include_transition=True,min_beats=MIN_BEATS_STATE)
                nearest=int(np.argmin(abs(center-bounds)))
                if feat:rows.append({"subject":sid,"start":center-half,"center":center,
                    "clip":int(c["clip"]),"label":0,"time_fraction":center/VIDEO_TOTAL_S,
                    "nearest_switch":int(switch_ids[nearest]),
                    "distance_to_boundary":float(abs(center-bounds[nearest])),**feat})
    return pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)


def transition_weights(frame):
    # Each participant contributes equal total weight to each available class.
    counts=frame.groupby(["subject","label"])["label"].transform("size").to_numpy()
    w=1.0/counts
    return w/np.mean(w)

def transition_cluster_cis(frame,probability,predicted,n=N_BOOT,seed=SEED):
    temp=frame[["subject","label"]].copy();temp["p"]=probability;temp["c"]=predicted
    rng=np.random.default_rng(seed);subs=temp["subject"].unique();rows=[]
    for _ in range(n):
        chosen=rng.choice(subs,len(subs),replace=True);parts=[temp[temp["subject"]==s] for s in chosen]
        z=pd.concat(parts,ignore_index=True)
        if z["label"].nunique()==2:
            rows.append(prediction_metrics(z["label"],z["p"],z["c"]))
    boot=pd.DataFrame(rows);out={}
    for metric in ["roc_auc","balanced_accuracy","pr_auc","brier"]:
        out[f"{metric}_ci_lo"]=boot[metric].quantile(.025)
        out[f"{metric}_ci_hi"]=boot[metric].quantile(.975)
    return out

def transition_loso(frame,features,model=None):
    X=frame[features];y=frame["label"].to_numpy(int);g=frame["subject"].to_numpy();p=np.full(len(frame),np.nan)
    predicted=np.full(len(frame),-1,int);folds=[]
    if model is None:
        model=LogisticRegression(max_iter=3000,random_state=SEED)
    for fold,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,g)):
        inner_frame=frame.iloc[tr].reset_index(drop=True);ix=inner_frame[features]
        iy=inner_frame["label"].to_numpy(int);ig=inner_frame["subject"].to_numpy();oof=np.full(len(inner_frame),np.nan)
        for a,b in GroupKFold(n_splits=min(5,len(np.unique(ig)))).split(ix,iy,ig):
            m=base_pipeline(clone(model));m.fit(ix.iloc[a],iy[a],clf__sample_weight=transition_weights(inner_frame.iloc[a]));oof[b]=m.predict_proba(ix.iloc[b])[:,1]
        cal_intercept,cal_slope=fit_probability_calibrator(iy,oof)
        inner_calibrated=apply_probability_calibrator(oof,cal_intercept,cal_slope)
        threshold=choose_threshold(iy,inner_calibrated)
        pipe=base_pipeline(clone(model));pipe.fit(X.iloc[tr],y[tr],clf__sample_weight=transition_weights(frame.iloc[tr])) # natural held-out prevalence
        raw=pipe.predict_proba(X.iloc[te])[:,1]
        p[te]=apply_probability_calibrator(raw,cal_intercept,cal_slope)
        predicted[te]=(p[te]>=threshold).astype(int)
        folds.append({"fold":fold,"test_subject":g[te][0],"inner_threshold":threshold,
                      "calibration_intercept":cal_intercept,"calibration_slope":cal_slope})
    return p,predicted,pd.DataFrame(folds)

def transition_auc_loso(frame,features,labels):
    X=frame[features];y=np.asarray(labels,int);g=frame["subject"].to_numpy();p=np.full(len(frame),np.nan)
    model=LogisticRegression(max_iter=3000,random_state=SEED)
    weighted_frame=frame.copy();weighted_frame["label"]=y
    for tr,te in LeaveOneGroupOut().split(X,y,g):
        pipe=base_pipeline(clone(model));pipe.fit(X.iloc[tr],y[tr],clf__sample_weight=transition_weights(weighted_frame.iloc[tr]));p[te]=pipe.predict_proba(X.iloc[te])[:,1]
    return roc_auc_score(y,p)

def transition_circular_null(frame,features,n_perm=N_TRANSITION_PERM,seed=SEED):
    observed=transition_auc_loso(frame,features,frame["label"]);rng=np.random.default_rng(seed);null=[]
    subject_indices={sid:g.sort_values("center").index.to_numpy() for sid,g in frame.groupby("subject")}
    for _ in range(n_perm):
        y=frame["label"].to_numpy(int).copy()
        for idx in subject_indices.values():
            shift=int(rng.integers(1,len(idx)));y[idx]=np.roll(y[idx],shift)
        null.append(transition_auc_loso(frame,features,y))
    null=np.asarray(null)
    return observed,null,float((np.sum(null>=observed)+1)/(len(null)+1))

def leave_one_boundary_out(frame,features):
    rows=[];model=LogisticRegression(max_iter=3000,random_state=SEED)
    for boundary in sorted(frame["nearest_switch"].unique()):
        tr=frame["nearest_switch"].to_numpy()!=boundary;te=~tr
        if frame.loc[te,"label"].nunique()<2 or frame.loc[tr,"label"].nunique()<2:continue
        train=frame.loc[tr].reset_index(drop=True);inner_prob=np.full(len(train),np.nan)
        iy=train["label"].to_numpy(int);ig=train["subject"].to_numpy();ix=train[features]
        for a,b in GroupKFold(n_splits=min(5,len(np.unique(ig)))).split(ix,iy,ig):
            m=base_pipeline(clone(model));m.fit(ix.iloc[a],iy[a],clf__sample_weight=transition_weights(train.iloc[a]));inner_prob[b]=m.predict_proba(ix.iloc[b])[:,1]
        cal_intercept,cal_slope=fit_probability_calibrator(iy,inner_prob)
        inner_calibrated=apply_probability_calibrator(inner_prob,cal_intercept,cal_slope)
        threshold=choose_threshold(iy,inner_calibrated)
        final=base_pipeline(clone(model));final.fit(frame.loc[tr,features],frame.loc[tr,"label"],clf__sample_weight=transition_weights(frame.loc[tr]))
        raw=final.predict_proba(frame.loc[te,features])[:,1]
        pp=apply_probability_calibrator(raw,cal_intercept,cal_slope);cc=(pp>=threshold).astype(int)
        rows.append({"held_boundary":boundary,**prediction_metrics(frame.loc[te,"label"],pp,cc)})
    return pd.DataFrame(rows)

transition_tables=[]
transition_control_rows=[]
for direction in ["NAI->AI","AI->NAI"]:
    d=build_transition_dataset(direction);p,c,folds=transition_loso(d,TRANSITION_FEATS)
    d["probability"]=p;d["predicted_class"]=c
    safe=direction.replace(">","to")
    d.to_csv(Path(OUTPUT_DIR)/f"transition_{safe}_predictions.csv",index=False)
    folds.to_csv(Path(OUTPUT_DIR)/f"Table_S9_{safe}_thresholds.csv",index=False)
    transition_tables.append({"direction":direction,"status":"thesis_RQ3",
                              **prediction_metrics(d["label"],p,c),
                              **transition_cluster_cis(d,p,c)})

    # Time-only diagnostic: unlike the submitted paper, this is not a matched-boundary control
    # and is not used to calibrate the physiological classifier.
    time_model=RandomForestClassifier(n_estimators=500,max_depth=4,
        random_state=SEED,n_jobs=-1)
    cp,cc,_=transition_loso(d,["time_fraction"],time_model)
    transition_control_rows.append({"direction":direction,"control":"time_only_diagnostic",
        **prediction_metrics(d["label"],cp,cc),**transition_cluster_cis(d,cp,cc)})
    boundary=leave_one_boundary_out(d,TRANSITION_FEATS)
    boundary.insert(0,"direction",direction)
    boundary.to_csv(Path(OUTPUT_DIR)/f"Table_S9_{safe}_leave_one_boundary.csv",index=False)
    observed,null,perm_p=transition_circular_null(d,TRANSITION_FEATS)
    pd.DataFrame({"null_auc":null}).to_csv(Path(OUTPUT_DIR)/f"Table_S9_{safe}_circular_null.csv",index=False)
    transition_control_rows.append({"direction":direction,"control":"physiology_circular_test",
                                    "n":len(d),"prevalence":d["label"].mean(),"roc_auc":observed,
                                    "permutation_p":perm_p})
transition_results=pd.DataFrame(transition_tables)
transition_results.to_csv(Path(OUTPUT_DIR)/"Table_S9_transition_results.csv",index=False)
pd.DataFrame(transition_control_rows).to_csv(Path(OUTPUT_DIR)/"Table_S9c_transition_controls.csv",index=False)
display(transition_results.round(3))
display(pd.DataFrame(transition_control_rows).round(3))

# %% Notebook code cell 32
def tsfresh_state_design(frame):
    records=[];meta=[]
    for wid,r in frame.reset_index(drop=True).iterrows():
        d=CACHE[r["subject"]];m=(d["t"]>=r["start"])&(d["t"]<r["end"])
        if m.sum()<MIN_BEATS_STATE:continue
        grid=np.arange(r["start"],r["end"],.25)
        records.append(pd.DataFrame({"id":wid,"time":np.arange(len(grid)),
            "hr":np.interp(grid,d["t"][m],d["hr"][m]),"nn":np.interp(grid,d["t"][m],d["nn"][m])}))
        meta.append({"id":wid,"subject":r["subject"],"clip":r["clip"],"label":r["ai"]})
    long=pd.concat(records,ignore_index=True);meta=pd.DataFrame(meta)
    feats=extract_features(long,column_id="id",column_sort="time",
        default_fc_parameters=EfficientFCParameters(),disable_progressbar=True,n_jobs=0)
    # Deliberately no global tsfresh.impute here.
    feats=feats.reindex(meta.id).reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
    return pd.concat([meta.reset_index(drop=True),feats],axis=1)

def tsfresh_transition_design(frame):
    records=[];meta=[]
    for wid,r in frame.reset_index(drop=True).iterrows():
        d=CACHE[r["subject"]];end=r["start"]+WINDOW_S;m=(d["t"]>=r["start"])&(d["t"]<end)
        if m.sum()<MIN_BEATS_STATE:continue
        grid=np.arange(r["start"],end,.25)
        records.append(pd.DataFrame({"id":wid,"time":np.arange(len(grid)),
            "hr":np.interp(grid,d["t"][m],d["hr"][m]),"nn":np.interp(grid,d["t"][m],d["nn"][m])}))
        meta.append({"id":wid,"subject":r["subject"],"clip":r["clip"],"label":r["label"]})
    long=pd.concat(records,ignore_index=True);meta=pd.DataFrame(meta)
    feats=extract_features(long,column_id="id",column_sort="time",
        default_fc_parameters=EfficientFCParameters(),disable_progressbar=True,n_jobs=0)
    feats=feats.reindex(meta.id).reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
    return pd.concat([meta.reset_index(drop=True),feats],axis=1)

def tsfresh_nested_loso(design):
    feature_cols=[c for c in design if c not in ["id","subject","clip","label"]]
    X=design[feature_cols];y=design["label"].to_numpy(int);g=design["subject"].to_numpy()
    p=np.full(len(design),np.nan);predicted=np.full(len(design),-1,int)
    selected=[]
    for outer,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,g)):
        inner=GroupKFold(n_splits=min(5,len(np.unique(g[tr]))));candidates={
          "LogReg":LogisticRegression(max_iter=3000,class_weight="balanced",random_state=SEED),
          "SVM":SVC(probability=True,class_weight="balanced",random_state=SEED)}
        oof_by_model={k:np.full(len(tr),np.nan) for k in candidates}
        for itr,iva in inner.split(X.iloc[tr],y[tr],g[tr]):
            a,b=tr[itr],tr[iva]
            for name,mdl in candidates.items():
                pipe=Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
                               ("select",SelectKBest(f_classif,k=min(TSFRESH_K,len(feature_cols)))),
                               ("scale",StandardScaler()),("clf",clone(mdl))])
                pipe.fit(X.iloc[a],y[a]);oof_by_model[name][iva]=pipe.predict_proba(X.iloc[b])[:,1]
        scores={name:roc_auc_score(y[tr],pv) for name,pv in oof_by_model.items()}
        chosen=max(scores,key=scores.get)
        cal_intercept,cal_slope=fit_probability_calibrator(y[tr],oof_by_model[chosen])
        inner_calibrated=apply_probability_calibrator(oof_by_model[chosen],cal_intercept,cal_slope)
        threshold=choose_threshold(y[tr],inner_calibrated)
        pipe=Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
                       ("select",SelectKBest(f_classif,k=min(TSFRESH_K,len(feature_cols)))),
                       ("scale",StandardScaler()),("clf",clone(candidates[chosen]))])
        pipe.fit(X.iloc[tr],y[tr]);raw=pipe.predict_proba(X.iloc[te])[:,1]
        p[te]=apply_probability_calibrator(raw,cal_intercept,cal_slope)
        predicted[te]=(p[te]>=threshold).astype(int)
        keep=np.array(feature_cols)[pipe.named_steps["select"].get_support()]
        selected.append({"fold":outer,"test_subject":g[te][0],"model":chosen,
                         "inner_threshold":threshold,"inner_auc":scores[chosen],
                         "calibration_intercept":cal_intercept,"calibration_slope":cal_slope,
                         "features":" | ".join(keep)})
    return p,predicted,pd.DataFrame(selected)

def selected_feature_frequency(selection):
    # Count fold-local selections; this is stability reporting, not global feature selection.
    rows=[]
    for _,r in selection.iterrows():
        for feature in str(r["features"]).split(" | "):
            if feature:rows.append({"feature":feature,"fold":r["fold"]})
    if not rows:return pd.DataFrame(columns=["feature","selected_folds","selection_fraction"])
    out=(pd.DataFrame(rows).groupby("feature").agg(selected_folds=("fold","nunique"))
         .sort_values("selected_folds",ascending=False).reset_index())
    out["selection_fraction"]=out["selected_folds"]/selection["fold"].nunique()
    return out

if RUN_TSFRESH:
    ts_design=tsfresh_state_design(state_df)
    ts_prob,ts_class,ts_selection=tsfresh_nested_loso(ts_design)
    ts_design["probability"]=ts_prob;ts_design["predicted_class"]=ts_class
    ts_threshold_map=ts_selection.set_index("test_subject")["inner_threshold"].to_dict()
    ts_design["threshold"]=ts_design["subject"].map(ts_threshold_map).astype(float)
    ts_clip=(ts_design.groupby(["subject","clip","label"],as_index=False)
        .agg(probability=("probability","median"),threshold=("threshold","median"))
        .assign(predicted_class=lambda x:(x["probability"]>=x["threshold"]).astype(int)))
    ts_state_result=pd.DataFrame([{"analysis":"tsfresh_state","status":"secondary_clip_level",
                                  **prediction_metrics(ts_clip["label"],ts_clip["probability"],ts_clip["predicted_class"]),
                                  **subject_auc_summary(ts_clip.rename(columns={"label":"ai"}),"probability")}])
    display(ts_state_result.round(3))
    ts_state_result.to_csv(Path(OUTPUT_DIR)/"Table_S7c_tsfresh_state_results.csv",index=False)
    ts_selection.to_csv(Path(OUTPUT_DIR)/"Table_S7b_tsfresh_nested_selection.csv",index=False)
    selected_feature_frequency(ts_selection).to_csv(
        Path(OUTPUT_DIR)/"Table_S7d_tsfresh_state_feature_stability.csv",index=False)
    ts_design[["id","subject","clip","label","probability","predicted_class"]].to_csv(
        Path(OUTPUT_DIR)/"tsfresh_outer_predictions.csv",index=False)
    ts_transition_summary=[]
    for direction in ["NAI->AI","AI->NAI"]:
        td=build_transition_dataset(direction)
        tdesign=tsfresh_transition_design(td)
        tp,tc,tsel=tsfresh_nested_loso(tdesign)
        tdesign["probability"]=tp;tdesign["predicted_class"]=tc
        ts_transition_summary.append({"direction":direction,"status":"exploratory",
                                      **prediction_metrics(tdesign["label"],tp,tc)})
        safe=direction.replace(">","to")
        tsel.to_csv(Path(OUTPUT_DIR)/f"Table_S9_tsfresh_{safe}_nested_selection.csv",index=False)
        selected_feature_frequency(tsel).to_csv(
            Path(OUTPUT_DIR)/f"Table_S9_tsfresh_{safe}_feature_stability.csv",index=False)
        tdesign[["id","subject","clip","label","probability","predicted_class"]].to_csv(
            Path(OUTPUT_DIR)/f"tsfresh_transition_{safe}_predictions.csv",index=False)
    pd.DataFrame(ts_transition_summary).to_csv(
        Path(OUTPUT_DIR)/"Table_S9b_tsfresh_transition_results.csv",index=False)

# %% Notebook code cell 34
# Descriptive coefficient stability for the fixed primary logistic model.
# Predictive performance remains based only on the outer held-out predictions.
coefficient_rows=[]
for held in state_df["subject"].unique():
    train=state_df[state_df["subject"]!=held]
    model=base_pipeline(clone(PRIMARY_MODEL))
    model.fit(train[CORE_FEATS],train["ai"],clf__sample_weight=clip_subject_weights(train))
    for feature,value in zip(CORE_FEATS,model.named_steps["clf"].coef_[0]):
        coefficient_rows.append({"held_out_subject":held,"feature":feature,
                                 "standardised_log_odds":value,"odds_ratio":np.exp(value)})
coefficient_stability=pd.DataFrame(coefficient_rows)
coefficient_summary=(coefficient_stability.groupby("feature",as_index=False)
    .agg(median_log_odds=("standardised_log_odds","median"),
         min_log_odds=("standardised_log_odds","min"),max_log_odds=("standardised_log_odds","max"),
         positive_fraction=("standardised_log_odds",lambda x:np.mean(np.asarray(x)>0))))
display(coefficient_summary)
coefficient_stability.to_csv(Path(OUTPUT_DIR)/"Table_S10_primary_coefficient_stability_folds.csv",index=False)
coefficient_summary.to_csv(Path(OUTPUT_DIR)/"Table_S10b_primary_coefficient_summary.csv",index=False)

# %% Notebook code cell 36
# Final audit bundle.
config={k:v for k,v in globals().items() if k in ["FS","ECG_COL","ECG_BAND","FILTER_ORDER",
    "WINDOW_S","STEP_S","GUARD_S","PURITY","MIN_BEATS_STATE","MIN_BEATS_TRANSITION",
    "MAX_CORRECTION_FRACTION","MAX_GAP_S","MIN_SUBJECT_NN","RECORDING_TOLERANCE_S",
    "PRIMARY_W","PRIMARY_FEATURES","CORE_FEATS","FULL_FEATS","TRANSITION_FEATS","SEED",
    "N_SIGNFLIP","N_BOOT","N_TRANSITION_PERM","TRANSITION_NEG_STEP_S","ANALYSIS_VERSION"]}
with open(Path(OUTPUT_DIR)/"analysis_config.json","w") as f: json.dump(config,f,indent=2)
with open(Path(OUTPUT_DIR)/"requirements_freeze.txt","w") as f:
    import subprocess
    f.write(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
print("V3 complete. Outputs written to",OUTPUT_DIR)

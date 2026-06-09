"""
Functions that were explored but didn't produce expected results.
Kept for reference — do not use in production code.
"""
import numpy as np
from scipy import stats
from signal_processing.fhr import compute_baseline, _apply_median_filter


def compute_baseline_sliding_window(signal, events, fs=4):
    """
    Sliding 10-minute window baseline, stepped by 1 minute.
    Did not produce expected results — use compute_baseline instead.
    """
    fhr_event_removed = remove_events(signal, events)
    n_samples         = len(fhr_event_removed)
    samples_per_min   = 60 * fs
    window_size       = 10 * samples_per_min
    step_size         = 1  * samples_per_min
    results           = []

    for start in range(0, n_samples - window_size, step_size):
        chunk = fhr_event_removed[start : start + window_size]
        if np.count_nonzero(~np.isnan(chunk)) >= (2 * 60 * fs):
            #results.append(round(np.nanmean(chunk) / 5) * 5)
            results.append(np.nanmean(chunk))
        else:
            results.append(np.nan)

    return results


def apply_mode_baseline(baselines, window_size_m=10):
    """
    Centered rolling mode over a window of minutes.
    Did not produce expected results — use compute_baseline instead.
    """
    baselines = np.array(baselines, copy=True)
    final_modes = []
    half_window = window_size_m // 2

    for i in range(0, len(baselines), 5):
        start  = max(0, i - half_window)
        end    = min(len(baselines), i + half_window + 1)
        window = [v for v in baselines[start:end] if not np.isnan(v)]
        if window:
            final_modes.append(stats.mode(window, keepdims=True).mode[0])
        else:
            final_modes.append(np.nan)

    return np.array(final_modes)


def compute_true_baseline_paper(fhr, window_size=1800, small_window_size=600, fs=4, alpha=8):
    """
    Baseline estimation from:
    'A novel cardiotocography fetal heart rate baseline estimation algorithm'.
    Uses a large window mean clamped within alpha bpm to produce a smoother baseline.
    """
    R_list = _compute_imaginary_baseline(fhr, window_size, fs)

    samples_per_small = small_window_size * fs
    n_small_windows   = int(np.ceil(len(fhr) / samples_per_small))
    ratio             = window_size // small_window_size

    baselines = []
    for j in range(n_small_windows):
        i = j // ratio
        if i >= len(R_list) or np.isnan(R_list[i]):
            baselines.append(np.nan)
            continue
        segment = fhr[j * samples_per_small : (j + 1) * samples_per_small]
        clamped = np.clip(segment, R_list[i] - alpha, R_list[i] + alpha)
        baselines.append(np.nanmean(clamped))

    return np.array(baselines)


def _compute_imaginary_baseline(fhr, window_size=1800, fs=4):
    """
    Computes the large-window mean used as reference in compute_true_baseline_paper.
    Private — use compute_true_baseline_paper instead.
    """
    samples_per_window = window_size * fs
    n_windows          = int(np.ceil(len(fhr) / samples_per_window))

    result = []
    for i in range(n_windows):
        segment = fhr[i * samples_per_window : (i + 1) * samples_per_window]
        if len(segment) > 0 and not np.all(np.isnan(segment)):
            result.append(np.nanmean(segment))
        else:
            result.append(np.nan)
    return np.array(result)

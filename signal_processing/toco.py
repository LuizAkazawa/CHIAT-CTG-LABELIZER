import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# ── Filtering ─────────────────────────────────────────────────────────────────


def apply_butterworth_filter(data, cutoff=0.05, fs=4.0, order=2):
    """
    Applies a low-pass Butterworth filter to smooth the TOCO signal.
    cutoff: frequency above which signals are attenuated (Hz)
    fs:     sampling frequency (Hz) — 4 Hz means 1 sample per 0.25s
    order:  steepness of the filter rolloff
    """
    nyquist = 0.5 * fs
    normal_cut = cutoff / nyquist
    b, a = butter(order, normal_cut, btype="low", analog=False)
    return filtfilt(b, a, data)


# ── Segmentation ──────────────────────────────────────────────────────────────


def get_toco_segments(raw_toco, toco_filtered, fs=4.0):
    """
    Splits the TOCO signal at calibration jumps (sudden ≥10 mmHg steps).
    Each segment gets its own baseline (10th percentile of filtered signal).
    Segments shorter than 2 minutes are ignored.
    """
    segments = []
    last_idx = 0
    min_seg_len = 120 * fs
    total_len = len(raw_toco)

    for i in range(1, total_len):
        is_jump = abs(raw_toco[i] - raw_toco[i - 1]) >= 10
        is_long_enough = (i - last_idx) >= min_seg_len

        if is_jump and is_long_enough:
            segment_data = toco_filtered[last_idx:i]
            if len(segment_data) > 0:
                segments.append(_make_segment(last_idx, i, segment_data, fs))
            last_idx = i

    # Last segment
    final_data = toco_filtered[last_idx:]
    if len(final_data) > 0:
        segments.append(_make_segment(last_idx, total_len - 1, final_data, fs))

    return segments




def _make_segment(start, end, data, fs):
    """Builds a segment dict. Private — used by get_toco_segments."""
    return {
        "indices": (start, end),
        "time_seconds": (start / fs, end / fs),
        "baseline": np.nanpercentile(data, 10),
    }


# ── Contraction detection ─────────────────────────────────────────────────────


def find_contractions(toco_filtered, segments, threshold=15, fs=4.0, time_threshold=2):
    """
    Detects uterine contractions within each TOCO segment.
    Steps:
      1. Find local peaks above threshold
      2. Merge close peaks / split at deep valleys (50% rule)
      3. Find start/end boundaries around each peak group
      4. Resolve any overlapping boundaries
    """
    toco_filtered = np.array(toco_filtered, copy=True)
    contractions = []
    peaks_save = []

    starts_seg = np.array([d["indices"][0] for d in segments])
    ends_seg = np.array([d["indices"][1] for d in segments])

    peaks, seg_thresholds = _find_valid_peaks(toco_filtered, segments)
    groups = _group_peaks(toco_filtered, peaks, starts_seg, ends_seg, seg_thresholds)

    seg_contractions = []
    for group in groups:
        start_peak = group[0]
        end_peak = group[1]

        result = _find_start_end(
            toco_filtered,
            starts_seg,
            ends_seg,
            seg_thresholds,
            time_threshold,
            start_peak,
            end_peak,
            fs,
        )
        if result is None:
            continue
        start, end = result

        duration_s = (end - start) / fs

        seg_contractions.append(
            {
                "start_seconds": start / fs,
                "end_seconds": end / fs,
                "start_idx": start,
                "end_idx": end,
                "duration": duration_s,
                # "peak_amplitude": float(np.max(toco_filtered[start:end]) - baseline),
                "peak_s": start_peak / fs,
            }
        )

    seg_contractions = _resolve_overlaps(seg_contractions, toco_filtered, fs)
    contractions.extend(seg_contractions)
    # print(seg_contractions)

    return contractions


# get all the values that is part of a contraction, not just the values inside the segment
def _find_valid_peaks(toco_filtered, segments, fs=4.0):
    """
    Finds valid contraction peaks using a segment-specific variable threshold.
    """
    valid_peaks = []
    peaks_scipy, _ = find_peaks(toco_filtered)

    for seg in segments:
        start, end = seg["indices"]
        baseline = seg["baseline"]

        peaks_in_segment = peaks_scipy[(peaks_scipy >= start) & (peaks_scipy < end)] #& (toco_filtered[peaks_scipy] >= baseline + 5)

        if len(peaks_in_segment) == 0:
            continue

        #mean_peak_val = np.mean(toco_filtered[peaks_in_segment])
        mean_peak_val = np.mean(toco_filtered[peaks_in_segment] - baseline)

        threshold = mean_peak_val #nao me agrada
        if threshold >= 15:
            threshold = 15 
        elif threshold <= 8:
            threshold = 10
        #print(threshold, start // 4)

        seg["lower_threshold"] = threshold

        for p in peaks_in_segment:
            if toco_filtered[p] >= baseline + threshold:
                valid_peaks.append(p)
    #print(segments)

    return valid_peaks, segments


def _group_peaks(toco_filtered, peaks, starts_segs, ends_segs, segments, percentage_threshold=0.5, fs=4.0):
    """
    Groups adjacent peaks that belong to the same contraction.
    Splits into a new group if the valley between them drops below a certain threshold
    """
    if len(peaks) == 0:
        return []

    def get_baseline_thresholds(idx):
        mask = (starts_segs <= idx) & (ends_segs > idx)
        matching_indices = np.where(mask)[0]
        if len(matching_indices) > 0:
            return segments[matching_indices[0]]["baseline"], segments[matching_indices[0]]["lower_threshold"] * percentage_threshold
        return np.nanpercentile(toco_filtered, 10)

    groups = []
    i = 0

    while i < len(peaks):
        start_peak = peaks[i]
        end_peak = peaks[i]

        j = i + 1
        while j < len(peaks):
            next_peak = peaks[j]

            # Find the deepest valley between the current end_peak and the next_peak
            valley_min = np.min(toco_filtered[end_peak : next_peak + 1])
            baseline, threshold = get_baseline_thresholds(end_peak)

            peak_amp = toco_filtered[end_peak] - baseline

            drop_amount = toco_filtered[end_peak] - valley_min

            if (toco_filtered[next_peak] - valley_min >= threshold or toco_filtered[peaks[j - 1]] - valley_min >= threshold):
                break
            else:
                end_peak = next_peak
                j += 1

        groups.append([start_peak, end_peak])
        i = j
    #print("GROUPS: ", groups)
    #print()
    return groups


def _find_start_end(signal, starts, ends, segments, time_threshold, start_peak, end_peak, fs=4.0):
    """
    Walks left and right from peak_idx to find where the contraction
    begins and ends (signal returning close to baseline).
    Private — used by find_contractions.
    """
    # Search LEFT
    start_idx = start_peak
    mask = (starts <= start_peak) & (ends > start_peak)

    matching = [d for d, m in zip(segments, mask) if m]
    #    print(matching)

    if not matching:
        return

    baseline = matching[0]["baseline"]
    threshold = matching[0]["lower_threshold"]
    seg_start = max(0, int(matching[0]["indices"][0] - 60 * fs))
    seg_end = min(len(signal) - 1, int(matching[0]["indices"][1] + 60 * fs))

    peak_amp = signal[start_peak] - baseline

    while start_idx > seg_start:
        if signal[start_idx] <= baseline + (threshold/2):
            break
        start_idx -= 1

    # Search RIGHT
    end_idx = end_peak

    while end_idx < seg_end - 1:
        if signal[end_idx] <= baseline + (threshold/2):
            break
        end_idx += 1
    if start_idx != start_peak and end_idx != end_peak:
        start_idx = start_idx + np.argmin(signal[start_idx:start_peak])
        end_idx = end_peak + np.argmin(signal[end_peak : end_idx + 1])
    #print("start and end and START_PEAK: ", start_idx, end_idx, start_peak)
    return start_idx, end_idx


def _resolve_overlaps(seg_contractions, toco_filtered, fs):
    seg_contractions.sort(key=lambda x: x["start_idx"])
    resolved = []

    if not seg_contractions:
        return resolved

    resolved.append(seg_contractions[0])

    for k in range(1, len(seg_contractions)):
        prev = resolved[-1]
        curr = seg_contractions[k]

        if curr["start_idx"] >= prev["end_idx"]:
            # No overlap
            resolved.append(curr)
            continue

        prev_peak_idx = int(prev["peak_s"] * fs)
        curr_peak_idx = int(curr["peak_s"] * fs)

        if curr_peak_idx <= prev_peak_idx:
            continue
        else:
            search_start = prev_peak_idx
            search_end = curr_peak_idx
            segment = toco_filtered[search_start:search_end]

            split_point = search_start + np.argmin(segment)

            prev["end_idx"] = split_point
            prev["end_seconds"] = split_point / fs
            prev["duration"] = (split_point - prev["start_idx"]) / fs
            curr["start_idx"] = split_point
            curr["start_seconds"] = split_point / fs
            curr["duration"] = (curr["end_idx"] - split_point) / fs
            resolved.append(curr)

    return resolved

def split_into_20min_chunks(fhr_signal, filtered_toco, raw_toco, contractions_list, fs=4.0):
    """
    Splits a long TOCO signal and its detected contractions list into
    STRICTLY complete 20-minute windows (4800 samples each).
    Any trailing remainder shorter than 20 minutes is discarded.
    """
    chunk_duration_sec = 20 * 60  # 1200 seconds
    chunk_samples = int(chunk_duration_sec * fs)  # 4800 samples
    total_samples = len(filtered_toco)

    # FIX: Use floor division to only count full, complete 20-minute pieces
    num_chunks = total_samples // chunk_samples

    dataset_chunks = []

    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_samples
        end_sample = start_sample + chunk_samples  # Always exactly +4800

        filtered_toco_chunk = filtered_toco[start_sample : end_sample]
        raw_toco_chunk = raw_toco[start_sample : end_sample]
        fhr_chunk = fhr_signal[start_sample : end_sample]

        # 2. Gather contractions that fall completely or partially inside this window
        chunk_contractions = []
        for con in contractions_list:
            # Check if there is any intersection with the current chunk window
            if con["start_idx"] < end_sample and con["end_idx"] >= start_sample:
                # Symmetrically trim boundaries if a contraction hits the edge of the 20-min mark
                rel_start = max(0, int(con["start_idx"] - start_sample))
                rel_end = min(chunk_samples - 1, int(con["end_idx"] - start_sample))

                adjusted_contraction = {
                    "start_idx": rel_start,
                    "end_idx": rel_end,
                    "start_seconds": float(rel_start / fs),
                    "end_seconds": float(rel_end / fs),
                    "duration": float((rel_end - rel_start) / fs),
                    "peak_s": float(max(0.0, con["peak_s"] - (start_sample / fs))),
                }
                chunk_contractions.append(adjusted_contraction)

        # 3. Append compiled chunk data
        dataset_chunks.append(
            {
                "chunk_index": chunk_idx,
                "filtered_toco_values": filtered_toco_chunk,  # Guaranteed length: 4800
                "raw_toco_values": raw_toco_chunk,  # Guaranteed length: 4800
                "fhr_values": fhr_chunk,
                "contractions": chunk_contractions,
            }
        )

    return dataset_chunks
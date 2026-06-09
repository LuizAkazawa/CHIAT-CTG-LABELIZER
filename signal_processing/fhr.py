import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ── Cleaning ─────────────────────────────────────────────────────────────────


def remove_outliers(signal, threshold=15, max_rejects=20):
    """
    Removes sample-to-sample jumps above threshold by setting them to NaN.
    Allows the baseline to shift if a jump is sustained (prevents erasing the whole signal).
    """
    clean = np.array(signal, copy=True)
    first_valid = np.where(~np.isnan(clean))[0]
    if len(first_valid) == 0:
        return clean

    last_valid_val = clean[first_valid[0]]
    reject_count = 0

    for i in range(first_valid[0] + 1, len(clean)):
        val = clean[i]
        if not np.isnan(val):
            if np.abs(val - last_valid_val) >= threshold and reject_count < max_rejects:
                clean[i] = np.nan
                reject_count += 1
            else:
                last_valid_val = val
                reject_count = 0
    return clean

def remove_outers(signal, valley_limit=50, peak_limit=240):
    clean = np.array(signal, copy=True)
    for i in range(len(clean)):
        if clean[i] < valley_limit or clean[i] > peak_limit:
            clean[i] = np.nan
    return clean


def remove_events(signal, events):
    signal = np.array(signal, copy=True)
    for ev in events:
        signal[ev["start_idx"] : ev["end_idx"]] = np.nan
    return signal


def fill_all_gaps(fhr_signal, sigma=3, reliability_threshold=15):
    """
    Fills short NaN gaps (< reliability_threshold seconds) with stochastic
    interpolation. Longer gaps are left as NaN (signal loss).
    """
    signal = np.array(fhr_signal, copy=True)
    n = len(signal)
    fs = 4
    i = 0

    while i < n:
        if np.isnan(signal[i]):
            gap_start = i
            while i < n and np.isnan(signal[i]):
                i += 1
            gap_end = i
            n_gap = gap_end - gap_start

            if n_gap < fs * reliability_threshold:
                y_start = signal[gap_start - 1] if gap_start > 0 else signal[gap_end]
                y_end = signal[gap_end] if gap_end < n else y_start
                signal[gap_start:gap_end] = _fill_gap_stochastic(
                    y_start, y_end, n_gap, sigma
                )
        else:
            i += 1
    return signal


def _fill_gap_stochastic(y_start, y_end, n_samples, sigma=0.4, amp_threshold=5):
    """
    Fills a gap with a linear trend plus a bounded random walk (Brownian bridge).
    Private — use fill_all_gaps instead.
    """
    np.random.seed(42)
    linear_trend = np.linspace(y_start, y_end, n_samples)
    noise = np.random.normal(0, sigma, n_samples)
    random_walk = np.cumsum(noise)

    bridge = np.linspace(random_walk[0], random_walk[-1], n_samples)
    stochastic = random_walk - bridge

    max_val = np.max(np.abs(stochastic))
    if max_val > amp_threshold:
        stochastic = stochastic * (amp_threshold / max_val)

    return linear_trend + stochastic


def apply_savgol_filter(fhr_filled, window_len=13, poly_order=3):
    smooth_fhr = np.full_like(fhr_filled, np.nan)
    valid_indices = np.where(~np.isnan(fhr_filled))[0]

    if len(valid_indices) > 0:
        splits = np.where(np.diff(valid_indices) > 1)[0] + 1
        segments = np.split(valid_indices, splits)

        for segment in segments:
            if len(segment) >= window_len:
                segment_data = fhr_filled[segment]
                smoothed_segment = savgol_filter(
                    segment_data, window_length=window_len, polyorder=poly_order
                )
                smooth_fhr[segment] = smoothed_segment
            else:
                smooth_fhr[segment] = fhr_filled[segment]
    return smooth_fhr




# ── Baseline ──────────────────────────────────────────────────────────────────

def compute_baseline(signal, window_size=600, fs=4):
    """
    Chunks the median-filtered signal into 10-minute windows
    and returns the mean of each window as the baseline.
    Zeros are ignored in the calculation.
    """
    clean_signal = np.array(signal, dtype=float, copy=True)
    clean_signal[clean_signal == 0] = np.nan

    smoothed = _apply_median_filter(clean_signal)
    
    samples_window = window_size * fs
    n_windows = len(clean_signal) // samples_window

    if n_windows == 0:
        return np.array([np.nanmean(smoothed)])

    baselines = []
    for i in range(n_windows):
        start = i * samples_window
        chunk = smoothed[start : start + samples_window]
        n_real = np.count_nonzero(~np.isnan(chunk))
        
        if n_real >= 120 * fs:
            baselines.append(np.nanmean(chunk))
        else:
            baselines.append(np.nan)

    return np.array(baselines)


def _apply_median_filter(signal, window_size=241):
    """
    Applies a median filter (~60s window) to suppress accels/decels
    before baseline estimation. Private — use compute_baseline instead.
    """
    padding = (window_size - 1) // 2
    padded = np.pad(signal, (padding, padding), mode="edge")
    output = np.zeros(len(signal))

    for i in range(len(signal)):
        output[i] = np.nanmedian(padded[i : i + window_size])
    return output


# ── Variability ───────────────────────────────────────────────────────────────


def compute_variability(signal, window_size=2400, quality_threshold=0.7):
    """
    Computes std deviation per 10-minute window as a measure of variability.
    Windows with too much missing data are marked NaN.
    """
    n_windows = len(signal) // window_size
    if n_windows == 0:
        return np.array([])

    variabilities = []
    for i in range(n_windows):
        start = i * window_size
        chunk = signal[start : start + window_size]
        quality = np.count_nonzero(~np.isnan(chunk)) / window_size
        if quality >= quality_threshold:
            variabilities.append(np.nanstd(chunk))
        else:
            variabilities.append(np.nan)

    return np.array(variabilities)


def compute_event_variability(segment, fs=4.0): 
    """
    Calculates the variability by removing the 'slope' of the deceleration.
    -> still testing
    """
    #try to do the idea -> if same direction we discard, otherwise we keep it. (can find direction -> bool var)
    going_up = False
    going_down = True
    nums = []
    for i in range(len(segment) - 1):
        if segment[i + 1] - segment[i] < 0 and not going_down:
            going_down = True
            going_up = False
            nums.append( (segment[i], segment[i + 1]))
        elif segment[i + 1] - segment[i] < 0 and going_down:
            nums.append(np.nan)
            continue
        
        if segment[i + 1] - segment[i] > 0 and not going_up:
            going_up = True
            going_down = False
            nums.append( (segment[i], segment[i + 1]))
        elif segment[i + 1] - segment[i] > 0 and going_up:
            nums.append(np.nan)
            continue
    #print()
    #print(nums)
    return 3



# ── Classification ────────────────────────────────────────────────────────────


def classify_baseline(baseline, rules):
    """Classifies each baseline value as Normal, Bradycardia, or Tachycardia."""
    rules = rules["baseline"]
    classifs = []
    for b in baseline:
        if np.isnan(b):
            classifs.append("N/A")
        elif b <= rules["severe_bradycardia"]["fhr_max"]:
            classifs.append("Severe_bradycardia")
        elif b <= rules["moderate_bradycardia"]["fhr_max"]:
            classifs.append("Moderate_bradycardia")
        elif b <= rules["mild_bradycardia"]["fhr_max"]:
            classifs.append("Mild_bradycardia")
        elif b <= rules["normal"]["fhr_max"]:
            classifs.append("Normal")
        elif b <= rules["moderate_tachycardia"]["fhr_max"]:
            classifs.append("Moderate_tachycardia")
        elif b >= rules["severe_tachycardia"]["fhr_min"]:
            classifs.append("Severe_tachycardia")
    return classifs


def classify_variability(variabilities, rules):
    """Classifies each variability value according to CTG rules."""
    rules = rules["variability"]
    classifs = []
    for v in variabilities:
        if np.isnan(v):
            classifs.append("N/A")
        elif v <= rules["absent"]["var_max"]:
            classifs.append("Absent")
        elif v <= rules["minimal"]["var_max"]:
            classifs.append("Minimal")
        elif v <= rules["normal"]["var_max"]:
            classifs.append("Normal")
        elif v >= rules["marked"]["var_min"]:
            classifs.append("Marked")
    return classifs


def classify_acceleration(rules, duration_sec, onset_to_peak, peak_amp):  # when acceleration has 2 peaks -> not fixed
    rules = rules["acceleration"]
    if duration_sec < rules["shoulder"]["duration_min"]:
        return False
    elif rules["prolonged_acceleration"]["duration_max"] < duration_sec <= rules["prolonged_acceleration"]["duration_max"] and onset_to_peak <= rules["prolonged_acceleration"]["diff_onset_to_peak_max"] and peak_amp >= rules["prolonged_acceleration"]["amp_min"]:
        return "Prolonged_Acceleration"
    elif rules["normal"]["duration_min"] < duration_sec < rules["normal"]["duration_max"] and onset_to_peak <= rules["normal"]["diff_onset_to_peak_max"] and peak_amp >= rules["normal"]["amp_min"] :
        return "Acceleration"
#    elif rules["shoulder"]["duration_min"] <= duration_sec <= rules["shoulder"]["duration_max"] and peak_amp >= rules["shoulder"]["amp_min"]:
#        return "Shoulder"
    return False


def is_peak(segment, idx_start, fs=4.0):
    w = int(2 * fs)

    loop_start = max(idx_start - w, 0)
    loop_end = min(idx_start + w, len(segment))

    for i in range(loop_start, loop_end):
        slice_start = max(0, i - w)
        slice_end = min(len(segment), i + w)

        if slice_start == slice_end:
            continue

        is_local_max = segment[i] == np.max(segment[slice_start:slice_end])

        if is_local_max:
            return True

    return False


def classify_deceleration(signal, event, rules, contractions, fs=4):
    rules = rules["deceleration"]

    onset_to_nadir = event["onset_to_nadir"]

    decel_start_sec = event["start_seconds"]
    decel_end_sec = event["end_seconds"]
    decel_nadir_sec = decel_start_sec + onset_to_nadir #the sec nadir happens

    start_idx = event["start_idx"]
    end_idx = event["end_idx"]

    duration_sec = event["duration"]

    segment = signal[start_idx:end_idx]

    if rules["prolonged"]["duration_min"] <= duration_sec <= rules["prolonged"]["duration_max"]:
        event["sub-type"] = "Prolonged Deceleration"

    if rules["variable"]["duration_min"] < duration_sec < rules["variable"]["duration_max"]:  # uniform or variable
        if onset_to_nadir >= rules["variable"]["onset_to_nadir_max"]:  # uniform -> if it's "smooth"
            rules_uniform = rules["uniform"]
            # not bad this solution
            matching_uc = None
            for uc in contractions: #looking if peak of UC is inside of a deceleration
                if uc["peak_s"] <= decel_end_sec and uc["peak_s"] >= decel_start_sec:
                    matching_uc = uc
                    break

            if matching_uc is not None: #need to check the peak of UC and the valley of FHR
                lag = abs(decel_nadir_sec - matching_uc["peak_s"]) #check the lag between FHR valley and UC peak

                if lag <= rules_uniform["early"]["lag_after_contraction_max"]:
                    event["sub-type"] = "Early Deceleration"
                else:
                    event["sub-type"] = "Late Deceleration"

        else:  # it's variable
            rules = rules["variable"]
            if event["attributes"]["has_initial_accel"] and event["attributes"]["has_terminal_accel"]:
                event["sub-type"] = "Variable Typical"
            
            elif event["attributes"]["has_initial_accel"] or event["attributes"]["has_terminal_accel"] or event["attributes"]["prolonged_sec_accel"] or event["attributes"]["slow_return"] or event["attributes"]["has_biphasic_shape"] or event["attributes"]["baseline_decrease"]:
                if event["nadir_value"] < 70 or event["amp_dec"] > 60 or event["duration"] > 60:
                    event["attributes"]["severity"] = "Severe"
                elif event["nadir_value"] >= 70 and event["amp_dec"] <= 60 and event["duration"] <= 60:
                    event["attributes"]["severity"] = "Moderate"
                else:
                    event["attributes"]["severity"] = "Undefined"
                event["sub-type"] = "Variable Atypical"
    else:
        event["sub-type"] = False
    return event



# ── Events ────────────────────────────────────────────────────────────────────

def resolve_fhr_overlaps(events):
    """
    Filters out invalid classifications (False) and then resolves overlaps 
    using a strict first-come, first-served rule.
    """
    valid_events = [e for e in events if e.get("sub-type") != False and e.get("sub-type") != ""]

    if not valid_events:
        return []

    sorted_events = sorted(valid_events, key=lambda x: x["start_idx"])

    resolved = [sorted_events[0]]

    for i in range(1, len(sorted_events)):
        curr = sorted_events[i]
        prev = resolved[-1]

        if curr["start_idx"] < prev["end_idx"]:
            continue
        else:
            #print(curr["start_idx"]/4, curr["end_idx"]/4)
            resolved.append(curr)

    return resolved



def find_events(filled_signal, baselines, fs=4, window_size=2400, time_threshold=3, minimum_time_beyond=1): #NEED TO REFACTOR THIS METHOD
    # remove threshold test?
    events = []
    i = 0
    n = len(filled_signal)
    eventSample = 0
    threshold_test = 5

    while i < n:
        current_baseline = baselines[min(i // window_size, len(baselines) - 1)]
        diff = filled_signal[i] - current_baseline

        # if abs(diff) >= 10:
        if diff >= 10 or diff <= -15:
            eventSample += 1
        else:
            eventSample = 0

        if eventSample >= minimum_time_beyond * fs: # min time beyond the threshold to detect events in seconds
            trigger_idx = i
            is_acceleration = diff > 0

            # Search LEFT for start
            start_idx = trigger_idx
            time_over = 0
            while start_idx > 0:
                if np.isnan(filled_signal[start_idx]):
                    start_idx += 1
                    break
                if (is_acceleration and filled_signal[start_idx] <= current_baseline + threshold_test):
                    time_over += 1
                if (not is_acceleration and filled_signal[start_idx] >= current_baseline - threshold_test):
                    time_over += 1
                if time_over >= time_threshold * fs:
                    break
                start_idx -= 1
            start_idx += time_over

            # Search RIGHT for end
            end_idx = trigger_idx
            time_over = 0
            while end_idx < n - 1:
                if np.isnan(filled_signal[end_idx]):
                    end_idx -= 1
                    break
                if is_acceleration and filled_signal[end_idx] <= current_baseline + threshold_test:
                    time_over += 1
                if not is_acceleration and filled_signal[end_idx] >= current_baseline - threshold_test:
                    time_over += 1
                if time_over >= time_threshold * fs:
                    break
                end_idx += 1
            end_idx -= time_over

            segment = filled_signal[start_idx:end_idx]
            duration_sec = (end_idx - start_idx) / fs
            max_amp = np.max(np.abs(segment - current_baseline))
            amp_dec = np.abs(np.min(segment - current_baseline))


            if is_acceleration:
                peak_idx = np.argmax(segment)
                onset_to_peak = peak_idx / fs
                events.append(
                    {
                        "type": "acceleration",
                        "sub-type": "",
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "start_seconds": start_idx / fs,
                        "end_seconds": end_idx / fs,
                        "peak_idx": peak_idx,
                        "onset_to_peak": onset_to_peak, 
                        "duration": duration_sec,
                        "max_amplitude": max_amp,
                    }
                )

            else:
                nadir_idx = start_idx + np.argmin(segment)
                onset_to_nadir = (nadir_idx - start_idx) / fs
                nadir_to_baseline = (end_idx - nadir_idx) / fs
                event_variability = compute_event_variability(filled_signal[start_idx : end_idx])
                baseline_after = np.round(compute_baseline(filled_signal[end_idx : end_idx + 600 * fs])) * 5
                nadir_value = filled_signal[nadir_idx]
                events.append(
                    {
                        "type": "deceleration",
                        "sub-type": "",
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "start_seconds": start_idx / fs,
                        "end_seconds": end_idx / fs,
                        "nadir_idx": nadir_idx,
                        "onset_to_nadir": onset_to_nadir, #slope
                        "nadir_to_baseline": nadir_to_baseline, #return to baseline
                        "duration": duration_sec, #duration
                        "amp_dec": amp_dec,
                        "nadir_value": nadir_value, 
                        "attributes": {
                            "slope": 0 if onset_to_nadir < 30 else 1, #0 -> sudden  (<30) ///  1 -> slow (>=30)
                            "duration_class": 0 if duration_sec < 120 else 1, # 0 -> short accel /// 1 -> prolonged accel
                            "has_residual_zone": False,
                            "has_initial_accel": False,
                            "has_terminal_accel": False,
                            "prolonged_sec_accel": False,
                            "slow_return" : False if nadir_to_baseline < 30 else True, #False -> fast /// 1 -> slow
                            "has_biphasic_shape": False,
                            "baseline_decrease": False if baseline_after >= current_baseline else True,
                            "absent_variability": False if event_variability >= 2 else True,
                            "severity": "Undefined"
                        }
                    }
                )
                # print(amp_dec)

            i = end_idx + 1
            eventSample = 0
        else:
            i += 1

    return events


def filter_shoulders(rules, events):
    shoulder_rules = rules["acceleration"]["shoulder"]
    decelerations = [
        e for e in events if e.get("type") == "deceleration" and e.get("sub-type")
    ]
    
    filtered_events = []

    for event in events:
        if event.get("type") == "acceleration":
            window_start = event["start_seconds"] - shoulder_rules["timing_relative_to_decel"]["initial_max_gap"]
            window_end = event["end_seconds"] + shoulder_rules["timing_relative_to_decel"]["terminal_max_gap"]
            
            was_appended = False

            for decel in decelerations:
                has_nearby_decel = (
                    decel["start_seconds"] <= window_end
                    and decel["end_seconds"] >= window_start
                )

                if has_nearby_decel:
                    # Adjust boundaries to touch the deceleration
                    if decel["start_seconds"] > event["end_seconds"]:
                        event["end_seconds"] = decel["start_seconds"]
                        
                    elif decel["end_seconds"] < event["start_seconds"]:
                        event["start_seconds"] = decel["end_seconds"]

                    filtered_events.append(event)
                    was_appended = True
                    break

            # Only append as "Normal" if it hasn't already been handled by the deceleration logic!
            if not was_appended and event.get("sub-type") == "Acceleration":
                filtered_events.append(event)
                
        else:
            # It's a deceleration or other event, just append it normally
            filtered_events.append(event)

    return filtered_events


def classify_events(signal, events, rules, contractions, fs=4):
    evs = np.array(events, copy=True)
    shoulder_rules = rules["acceleration"]["shoulder"]["timing_relative_to_decel"]

    for e in evs:
        is_lackingInfo = False
        fds_min = max(e["start_idx"] - (30 * fs), 0)
        fds_max = min(len(signal), e["end_idx"] + (30 * fs))
        window_size = fds_max - fds_min

        quality = np.count_nonzero(~np.isnan(signal[fds_min:fds_max])) / (window_size)
        #print("quality:", quality)

        if quality < 0.8:
            is_lackingInfo = True
        if e["type"] == "acceleration":
            if is_lackingInfo:
                e["sub-type"] = f"Acceleration : {quality:.2f}"
                continue
            e["sub-type"] = classify_acceleration(
                rules,
                e["duration"],
                e["onset_to_peak"],
                e["max_amplitude"],
            )
        elif e["type"] == "deceleration":
            # start_idx = max(0, e["start_idx"])
            # end_idx = min(len(signal), e["end_idx"])
            lower_bound = max(0, e["start_seconds"] - shoulder_rules["initial_max_gap"])  # 10 seconds -> threshold to accept a shoulder with deceleration
            upper_bound = min(e["end_seconds"] + shoulder_rules["terminal_max_gap"], len(signal) / 4)

            has_initial_accel= any(
                #'Shoulder' in str(other.get('sub-type', ''))
                "acceleration" in str(other.get("type", ""))
                for other in evs
                if other["type"] == "acceleration"
                and lower_bound <= other["end_seconds"] <= e["start_seconds"]
            )
            if has_initial_accel:
                e["attributes"]["has_initial_accel"] = True


            has_terminal_accel = any(
                "acceleration" in str(other.get("type", ""))
                for other in evs
                if other["type"] == "acceleration"
                and e["end_seconds"] <= other["start_seconds"] <= upper_bound
            )
            if has_terminal_accel:
                e["attributes"]["has_terminal_accel"] = True

            e = classify_deceleration(
                signal,
                e,
                rules,
                contractions,
            )
            if is_lackingInfo and e["sub-type"] != False:
                e["sub-type"] = f"Deceleration"
            #if e["sub-type"] != False:
                #print(e)
                #print()

    return evs

import json
import numpy as np
from PyQt6.QtWidgets import QFileDialog
import os

from signal_processing.io import read_cts_file, trim_by_derivative
from signal_processing.fhr import (
    apply_savgol_filter,
    classify_baseline,
    classify_events,
    classify_variability,
    compute_baseline,
    compute_variability,
    fill_all_gaps,
    filter_shoulders,
    find_events,
    remove_events,
    remove_outliers,
    remove_outers,
    resolve_fhr_overlaps
)
from signal_processing.toco import (
    apply_butterworth_filter,
    get_toco_segments,
    find_contractions,
    split_into_20min_chunks
)


RULES_PATH = "structured_description-2.json"


class DataLoader:
    def __init__(self, main_window, plot_area, sidebar):
        self.main_window = main_window  
        self.plot_area = plot_area
        self.sidebar = sidebar
        self.ctg_rules = self._load_rules()

    # ── Rules ────────────────────────────────────────────────────────────────

    def _load_rules(self):
        try:
            with open(RULES_PATH, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {RULES_PATH} not found.")
            return {}

    # ── Entry point ──────────────────────────────────────────────────────────

    # ── Entry point ──────────────────────────────────────────────────────────

    def load_data(self):
        """Called by the manual 'Load CTS File' button."""
        if not self.ctg_rules:
            return

        filepath = self._pick_file()
        if not filepath:
            return
            
        self.load_specific_file(filepath)

    def load_specific_file(self, filepath):
        """Loads a file directly, bypassing the dialog picker."""
        if not self.ctg_rules:
            return
            
        if not os.path.exists(filepath):
            print(f"Error: File not found {filepath}")
            return

        filename = os.path.basename(filepath)
        self.main_window.setWindowTitle(f"Interactive CTG Navigator - {filename}")

        # Store the current filepath so the viewer knows what file is being corrected!
        self.current_filepath = filepath

        # Read & pre-process
        df = read_cts_file(filepath)
        fhr_raw, fhr_clean, fhr_filled, smooth_fhr, raw_toco, filtered_toco, fetal_movs, time_data = self._pre_process(df)

        if len(fhr_clean) == 0:
            print("Error: no FHR data found.")
            return

        # Signal processing
        toco_segments = get_toco_segments(raw_toco, filtered_toco)

        raw_contractions = find_contractions(filtered_toco, toco_segments, threshold=15)
        self.fhr_windows = self._process_fhr(fhr_clean, fhr_filled, smooth_fhr, raw_contractions)
        self.fhr_events = self._process_events(fhr_filled, self.fhr_windows, raw_contractions)
        self.toco_chunks = split_into_20min_chunks(fhr_raw, filtered_toco, raw_toco, raw_contractions, fs=4.0)

        # Map global references directly to the chunks
        self.contractions = []
        for chunk in self.toco_chunks:
            for c in chunk["contractions"]:
                self.contractions.append(c)

        # Update plots
        self.plot_area.clear_annotations()
        self.plot_area.update_signals(time_data, fhr_raw, raw_toco)

        self.plot_area.register_callback_click(
            callback_funcao=self.add_manual_contraction, 
            callback_delete=self.remove_contraction_toco,
            cb_fhr_add=self.add_fhr_event,
            cb_fhr_del=self.remove_fhr_event,
            cb_status_update=self.update_fhr_status,
            cb_baseline_add=self.add_baseline_window,     
            cb_baseline_del=self.remove_baseline_window   
        )

        self._draw_all(
            time_data, toco_segments, 
            self.fhr_windows,
            self.toco_chunks, self.fhr_events, fetal_movs
        )
        self.plot_area.reset_view()

    # ── File picking ─────────────────────────────────────────────────────────

    def _pick_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            None, "Open CTS File", "", "CTS Files (*.cts)"
        )
        return filepath

    # ── Pre-processing ───────────────────────────────────────────────────────

    def _pre_process(self, df):
        start_index   = trim_by_derivative(df["TOCO"].values, df["FHR1"].values)
        #fhr_raw       = df['FHR1'].where(df['SIG1'] == 2).values[start_index:]
        fhr_raw       = df['FHR1'].values[start_index:]
        fhr_raw = remove_outers(fhr_raw)

        toco_series = df["TOCO"].where(df["TOCOSIG"] == 2)
        toco_series_interpolated = toco_series.interpolate(method='linear').ffill().bfill()
        raw_toco = toco_series_interpolated.values[start_index:]

        fetal_movs_raw = df['FMP'].where(df["TOCOSIG"] == 2).values[start_index:]
        fetal_movs     = np.where(fetal_movs_raw > 0)[0]

        filtered_toco = apply_butterworth_filter(raw_toco, cutoff=0.02, fs=4.0)

        if len(fhr_raw) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        fhr_clean  = remove_outliers(fhr_raw)
        fhr_filled = fill_all_gaps(fhr_clean)
        time_data  = np.arange(len(fhr_clean)) * 0.25

        smooth_fhr = apply_savgol_filter(fhr_filled, window_len=13, poly_order=3)



        return fhr_raw, fhr_clean, fhr_filled, smooth_fhr, raw_toco, filtered_toco, fetal_movs, time_data

    # ── Signal processing ────────────────────────────────────────────────────

    def _process_fhr(self, fhr_clean, fhr_filled, smooth_fhr, contractions, fs=4):
        missing_threshold = 120
        bases = compute_baseline(fhr_clean)
        rounded_bases = np.round(np.array(bases, dtype=float) / 5) * 5

        events = find_events(fhr_filled, rounded_bases)
        class_events = classify_events(fhr_filled, events, self.ctg_rules, contractions)
        resolved_events = resolve_fhr_overlaps(class_events)
        filtered_events = filter_shoulders(self.ctg_rules, resolved_events)

        clean_signal = remove_events(fhr_filled, filtered_events)
        remo_baselines = compute_baseline(clean_signal, 600)
        remo_baselines = np.round(np.array(remo_baselines, dtype=float) / 5) * 5

        true_baseline = np.array(rounded_bases, copy=True)
        # if has events for more than 2 minutes -> purple, else -> rouge /// if has more than 2 minutes of missing data -> rouge
        for i in range(len(true_baseline) - 1):
            start_idx = i * 600 * fs
            end_idx = min((i + 1) * 600 * fs, len(fhr_clean))
            segment = fhr_clean[start_idx:end_idx]
            events_interest = [
                min(event["end_idx"], end_idx) - max(event["start_idx"], start_idx)
                for event in events
                if event["start_idx"] < end_idx and event["end_idx"] > start_idx
            ]
            if np.count_nonzero(np.isnan(segment)) > missing_threshold * 4 * fs:
                true_baseline[i] = np.nan
            elif np.count_nonzero(np.isnan(segment)) > missing_threshold * fs:
                continue
            elif np.sum(events_interest) > missing_threshold * fs:
                true_baseline[i] = remo_baselines[i]

        # first_slide_bases = compute_baseline_sliding_window(fhr_filled, events)
        # sliding_baselines = apply_mode_baseline(first_slide_bases)
        # sliding_baselines = np.round(np.array(sliding_baselines, dtype=float) / 5) * 5

        bases_class = classify_baseline(true_baseline, self.ctg_rules)
        variabilities = compute_variability(fhr_filled)
        var_class = classify_variability(variabilities, self.ctg_rules)

        fhr_windows = []
        for i in range(len(true_baseline)):
            if np.isnan(true_baseline[i]):
                continue
            start_s = i * 600.0
            end_s = (i + 1) * 600.0
            
            fhr_windows.append({
                "type": "window",
                "start_seconds": float(start_s),
                "end_seconds": float(end_s),
                "start_idx": int(start_s * fs),
                "end_idx": int(end_s * fs),
                "baseline": float(true_baseline[i]),
                "base_class": bases_class[i],
                "variability": float(variabilities[i]) if i < len(variabilities) and not np.isnan(variabilities[i]) else 0.0,
                "var_class": var_class[i] if i < len(var_class) else "Undefined"
            })
            
        return fhr_windows

    def _process_events(self, fhr_filled, fhr_windows, contractions):
        rounded_bases = [w["baseline"] for w in fhr_windows] 
        if not rounded_bases:
            rounded_bases = [120.0] #just testing
        events = find_events(fhr_filled, rounded_bases)
        class_events = classify_events(fhr_filled, events, self.ctg_rules, contractions)
        resolved_events = resolve_fhr_overlaps(class_events)
        filtered_events = filter_shoulders(self.ctg_rules, resolved_events)
        #print(filtered_events)
        return filtered_events

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw_all(self, time_data, toco_segments, fhr_windows, toco_chunks, class_events, fetal_movs):

        sb = self.sidebar

        self.plot_area.draw_grid_lines(time_data[-1])
        self.plot_area.draw_toco_baselines(toco_segments)
        
        self.plot_area.draw_fhr_windows(fhr_windows) 

        for chunk in toco_chunks:
            chunk_idx = chunk.get("chunk_index", 0)
            chunk_offset = chunk_idx * 1200.0
            self.plot_area.draw_contractions(
                chunk["contractions"], 
                chunk_offset_seconds=chunk_offset,
                chunk_signal=chunk["raw_toco_values"] 
            )

        self.plot_area.draw_fetal_movs(fetal_movs)
        self.plot_area.draw_events(class_events)


    # ── Areas Creation ──────────────────────────────────────────────────────────────
    def add_manual_contraction(self, tempo_segundos):
        """Cria uma nova contração baseada no clique do usuário."""
        fs = 4.0
        duracao_padrao = 60.0

        chunk_idx = int(tempo_segundos // 1200)

        if not (hasattr(self, "toco_chunks") and chunk_idx < len(self.toco_chunks)):
            print(f"Error: chunk {chunk_idx} not found.")
            return

        relative_time_seconds = tempo_segundos % 1200
        start_rel_s = max(0.0, relative_time_seconds - (duracao_padrao / 2))
        end_rel_s = relative_time_seconds + (duracao_padrao / 2)

        start_idx = int(start_rel_s * fs)
        end_idx = int(end_rel_s * fs)

        # ── Find the true peak in the signal slice ──
        chunk_toco_signal = self.toco_chunks[chunk_idx]["raw_toco_values"]
        
        start_idx = max(0, start_idx)
        end_idx = min(len(chunk_toco_signal), end_idx)
        segment = chunk_toco_signal[start_idx:end_idx]
        
        if len(segment) > 0:
            local_peak_idx = np.argmax(segment)
            true_peak_rel_idx = start_idx + local_peak_idx
            true_peak_s = true_peak_rel_idx / fs
        else:
            # Fallback just in case the segment calculation goes out of bounds
            true_peak_s = relative_time_seconds

        new_contraction = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_seconds": float(start_rel_s),
            "end_seconds": float(end_rel_s),
            "duration": float(duracao_padrao),
            "peak_s": float(true_peak_s), 
        }

        # Append reference to both places cleanly
        self.toco_chunks[chunk_idx]["contractions"].append(new_contraction)
        self.contractions.append(new_contraction)

        chunk_offset = chunk_idx * 1200.0
        
        self.plot_area.draw_contractions(
            [new_contraction], chunk_offset_seconds=chunk_offset, chunk_signal=chunk_toco_signal
        )
        print(
            f"New contraction added on chunk: {chunk_idx}. Clicked at {relative_time_seconds:.2f}s, Peak snapped to {true_peak_s:.2f}s"
        )

    def remove_contraction_toco(self, meta):
        """Remove contraction from memory."""
        if meta in self.contractions:
            self.contractions.remove(meta)

        if hasattr(self, "toco_chunks"):
            for chunk in self.toco_chunks:
                if meta in chunk["contractions"]:
                    chunk["contractions"].remove(meta)
                    break 

        print(f"Contraction succesfully removed from both buffers.")


    def add_fhr_event(self, time_seconds, event_type="acceleration"):
        """Cria um novo evento FHR baseado no clique do usuário."""
        default_width = 45.0  # Default width for events
        start_s = max(0.0, time_seconds - (default_width / 2))
        end_s = time_seconds + (default_width / 2)

        # Define the sub-type dynamically based on the click
        sub_type = "Acceleration" if event_type == "acceleration" else "Deceleration"

        if event_type == "acceleration":
            new_event = {
                "type": event_type,
                "sub-type": sub_type,
                "start_seconds": float(start_s),
                "end_seconds": float(end_s),
                "start_idx": int(start_s * 4.0),
                "end_idx": int(end_s * 4.0)
            }
        
        else:
            new_event = {
                "type": event_type,
                "sub-type": sub_type,
                "start_seconds": float(start_s),
                "end_seconds": float(end_s),
                "start_idx": int(start_s * 4.0),
                "end_idx": int(end_s * 4.0),
                "attributes": {
                    "slope": 0,
                    "duration_class": 0, # 0 -> short accel /// 1 -> prolonged accel
                    "has_residual_zone": False,
                    "has_initial_accel": False,
                    "has_terminal_accel": False,
                    "prolonged_sec_accel": False,
                    "slow_return" : False, #False -> fast /// 1 -> slow
                    "has_biphasic_shape": False,
                    "baseline_decrease": False,
                    "absent_variability": False,
                    "severity": "Undefined"
                }
            }

        self.fhr_events.append(new_event)
        
        # Draw just this new event incrementally
        self.plot_area.draw_events([new_event])
        print(f"New FHR event ({event_type}) added manually in {time_seconds:.2f}s")


    def update_fhr_status(self, meta_dict, new_base_val, new_var_val):
        """Update the dictionary directly on memory."""
        
        meta_dict["baseline"] = float(new_base_val)
        meta_dict["variability"] = float(new_var_val)

        base_results = classify_baseline([new_base_val], self.ctg_rules)
        var_results = classify_variability([new_var_val], self.ctg_rules)

        new_base_class = base_results[0] if base_results else "Undefined"
        new_var_class = var_results[0] if var_results else "Undefined"

        meta_dict["base_class"] = new_base_class
        meta_dict["var_class"] = new_var_class
            
        print(f"Window dynamically reclassified: {new_base_val} -> {new_base_class}")
        return new_base_class, new_var_class

    def remove_fhr_event(self, meta):
        """Removes a FHR event from memory."""
        if meta in self.fhr_events:
            self.fhr_events.remove(meta)
            print("FHR event succesfully removed.")

    
    def add_baseline_window(self, time_sec):
        """Create a new 10min window with baseline/variability to be defined."""
        window_size = 600.0 
        
        start_s = max(0.0, time_sec - (window_size / 2))
        end_s = time_sec + (window_size / 2)
        
        new_window = {
            "type": "window",
            "start_seconds": float(start_s),
            "end_seconds": float(end_s),
            "start_idx": int(start_s * 4.0),
            "end_idx": int(end_s * 4.0),
            "baseline": 120.0,    # Safe default; user can immediately double-click to edit
            "base_class": "Normal",
            "variability": 5.0,   # Safe default
            "var_class": "Normal"
        }
        
        self.fhr_windows.append(new_window)
        self.plot_area.draw_fhr_windows([new_window])
        print(f"New Baseline Window added manually at {time_sec:.2f}s")
        
    def remove_baseline_window(self, meta):
        """Remove baseline window from memory."""
        if hasattr(self, 'fhr_windows') and meta in self.fhr_windows:
            self.fhr_windows.remove(meta)
            print("Baseline window successfully removed from memory.")
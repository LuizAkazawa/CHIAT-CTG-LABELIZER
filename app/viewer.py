# app/viewer.py
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget
import pyqtgraph as pg
import json
import numpy as np
import os
import math

from app.ui.sidebar import Sidebar
from app.ui.plots import PlotArea
from app.controllers.data_loader import DataLoader


def save_chunks_to_json(dataset_chunks, fhr_events, fhr_windows, toco_segments, output_filepath):
    """
    Saves the strict 20-minute signal chunks, calculated contraction metrics,
    and FHR events to a formatted JSON file for convenient hand-corrections.
    """
    serializable_chunks = []
    CHUNK_DURATION_S = 1200.0 

    for chunk in dataset_chunks:
        chunk_idx = int(chunk["chunk_index"])
        chunk_start_s = chunk_idx * CHUNK_DURATION_S
        chunk_end_s = (chunk_idx + 1) * CHUNK_DURATION_S

        filtered_toco_arr = chunk["filtered_toco_values"]
        raw_toco_arr = chunk["raw_toco_values"]
        raw_fhr_arr = chunk["raw_fhr_values"]
        smooth_fhr_arr = chunk["smooth_fhr_values"]

        filtered_toco_signal = [None if math.isnan(v) else int(round(v)) for v in filtered_toco_arr]
        raw_toco_signal = [None if math.isnan(v) else int(round(v)) for v in raw_toco_arr]
        raw_fhr_signal = [None if math.isnan(v) else int(round(v)) for v in raw_fhr_arr]
        smooth_fhr_signal = [None if math.isnan(v) else int(round(v)) for v in smooth_fhr_arr]

        # ── Formatted Contractions ──────────────────────────────────────────
        formatted_contractions = []
        for con in chunk["contractions"]:
            formatted_contractions.append(
                {
                    "start_idx": int(con["start_idx"]),
                    "end_idx": int(con["end_idx"]),
                    "start_seconds": float(con["start_seconds"]),
                    "end_seconds": float(con["end_seconds"]),
                    "duration_seconds": float(con["duration"]),
                    "peak_seconds": float(con["peak_s"]),
                }
            )

        # ── Formatted FHR Events ────────────────────────────────────────────
        formatted_fhr = []
        for ev in fhr_events:
            # Check if this event belongs to the current 20-minute chunk
            if chunk_start_s <= ev["start_seconds"] < chunk_end_s and ev["sub-type"] != False:
                formatted_fhr.append(
                    {
                        "type": str(ev.get("type", "")),
                        "sub_type": str(ev.get("sub-type", "")),
                        "start_idx": int(ev["start_idx"]),
                        "end_idx": int(ev["end_idx"]),
                        "start_seconds": float(ev["start_seconds"]),
                        "end_seconds": float(ev["end_seconds"]),
                        "attributes": ev.get("attributes", {}) 
                    }
                )

        formatted_fms = []
        for fm_idx in chunk.get("fetal_movs", []):
            formatted_fms.append({
                "idx": int(fm_idx),
                "seconds": float(fm_idx / 4.0)
            })

        serializable_chunks.append(
            {
                "chunk_index": chunk_idx,
                "filtered_toco_values": filtered_toco_signal,
                "raw_toco_values" : raw_toco_signal,
                "raw_fhr_values": raw_fhr_signal,
                "smooth_fhr_values": smooth_fhr_signal,
                "contractions": formatted_contractions,
                "fhr_events": formatted_fhr,  
                "fetal_movements": formatted_fms
            }
        )

    formatted_windows = []
    for win in fhr_windows:
        formatted_windows.append({
            "start_idx": int(win["start_idx"]),
            "end_idx": int(win["end_idx"]),
            "start_seconds": float(win["start_seconds"]),
            "end_seconds": float(win["end_seconds"]),
            "baseline": float(win["baseline"]),
            "base_class": str(win["base_class"]),
            "variability": float(win["variability"]),
            "var_class": str(win["var_class"])
        })

    formatted_toco_bases = []
    for seg in toco_segments:
        formatted_toco_bases.append({
            "start_idx": int(seg["indices"][0]),
            "end_idx": int(seg["indices"][1]),
            "start_seconds": float(seg["time_seconds"][0]),
            "end_seconds": float(seg["time_seconds"][1]),
            "baseline": float(seg["baseline"])
        })

    final_export_data = {
        "windows_info": formatted_windows,
        "toco_baselines": formatted_toco_bases,
        "chunks": serializable_chunks
    }

    # Write file out with clean indentation
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(final_export_data, f, indent=4, ensure_ascii=False)

    print(f"Successfully generated dataset JSON for corrections: {output_filepath}")

class CTGInteractiveViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive CTG Navigator")
        self.resize(1200, 600)

        # Build UI pieces
        self.sidebar = Sidebar()
        self.plot_area = PlotArea()
        self.data_loader = DataLoader(self, self.plot_area, self.sidebar)

        # Assemble layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.plot_area)

        # Connect sidebar actions to controller
        self.sidebar.load_requested.connect(self.data_loader.load_data)
        self.sidebar.save_requested.connect(self._save_corrections_to_json)
        self.sidebar.speed_changed.connect(self.plot_area.change_speed)
        self.sidebar.file_selected.connect(self.data_loader.load_specific_file)




    def _save_corrections_to_json(self):
        """Calls the save routine, dynamically passing the data stored in memory."""
        if hasattr(self.data_loader, "toco_chunks") and self.data_loader.toco_chunks:
            
            TARGET_SAVE_FOLDER = "Data/corrected_jsons"
            os.makedirs(TARGET_SAVE_FOLDER, exist_ok=True) 
            
            if hasattr(self.data_loader, "current_filepath") and self.data_loader.current_filepath:
                file_name = os.path.basename(self.data_loader.current_filepath)
                name_only, _ = os.path.splitext(file_name)
                output_path = os.path.join(TARGET_SAVE_FOLDER, f"{name_only}_chunks_corrected.json")
            else:
                output_path = os.path.join(TARGET_SAVE_FOLDER, "Num1_RData_chunks_corrected.json")

            current_fhr_events = getattr(self.data_loader, "fhr_events", [])
            current_fhr_windows = getattr(self.data_loader, "fhr_windows", [])
            current_toco_segments = getattr(self.data_loader, "toco_segments", [])

            # Execute save with the new parameter
            save_chunks_to_json(self.data_loader.toco_chunks, current_fhr_events, current_fhr_windows, current_toco_segments, output_path)
        else:
            print("Error: no data to save.")


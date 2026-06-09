# app/viewer.py
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget
import pyqtgraph as pg
import json
import numpy as np
import os

from app.ui.sidebar import Sidebar
from app.ui.plots import PlotArea
from app.controllers.data_loader import DataLoader


import json
import numpy as np
import os
import math  

def save_chunks_to_json(dataset_chunks, fhr_events, output_filepath):
    """
    Saves the strict 20-minute signal chunks, calculated contraction metrics,
    and FHR events to a formatted JSON file for convenient hand-corrections.
    """
    serializable_chunks = []
    
    # 20 minutes in seconds
    CHUNK_DURATION_S = 1200.0 

    for chunk in dataset_chunks:
        chunk_idx = int(chunk["chunk_index"])
        chunk_start_s = chunk_idx * CHUNK_DURATION_S
        chunk_end_s = (chunk_idx + 1) * CHUNK_DURATION_S

        # ── OPTIMIZED SIGNAL CONVERSION ──
        # Extract the raw numpy arrays directly
        toco_arr = chunk["filtered_toco_values"]
        fhr_arr = chunk["fhr_values"]

        # Fast list comprehension: 
        # 1. Checks for NaN using the highly optimized math.isnan
        # 2. Replaces NaNs with None (which becomes standard 'null' in JSON)
        # 3. Rounds valid floats to the nearest whole number and casts to int
        toco_signal_list = [None if math.isnan(v) else int(round(v)) for v in toco_arr]
        fhr_signal_list = [None if math.isnan(v) else int(round(v)) for v in fhr_arr]

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

        serializable_chunks.append(
            {
                "chunk_index": chunk_idx,
                "toco_values": toco_signal_list,
                "fhr_values": fhr_signal_list,
                "contractions": formatted_contractions,
                "fhr_events": formatted_fhr,  
            }
        )

    # Write file out with clean indentation
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, indent=4, ensure_ascii=False)

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
        """Chama a rotina de salvamento passando os dados que estão em memória de forma dinâmica."""
        # Check if loader has data loaded
        if hasattr(self.data_loader, "toco_chunks") and self.data_loader.toco_chunks:
            # Get the original file path loaded dynamically
            if hasattr(self.data_loader, "current_filepath") and self.data_loader.current_filepath:
                base_path, _ = os.path.splitext(self.data_loader.current_filepath)
                output_path = f"{base_path}_chunks_corrected.json"
            else:
                # Fallback path if path retrieval failed
                output_path = "Data/Num1_RData_chunks_corrected.json"

            # Safely fetch FHR events from the DataLoader (defaults to empty list if none exist)
            current_fhr_events = getattr(self.data_loader, "fhr_events", [])

            # Execute save with the new parameter
            save_chunks_to_json(self.data_loader.toco_chunks, current_fhr_events, output_path)
        else:
            print("Error: no data to save.")


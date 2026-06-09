# app/ui/plots.py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QApplication, QInputDialog, 
                             QDialog, QFormLayout, QComboBox, QDialogButtonBox,
                             QSpinBox, QDoubleSpinBox) # <-- Added SpinBoxes
from PyQt6.QtCore import Qt


# ── Pens & Colors ───────────────────────────────────────────────────────────

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

PEN_RAW_FHR        = pg.mkPen('b', width=1.5)
PEN_FILLED_FHR     = pg.mkPen('g', width=1.5)
PEN_SMOOTH_FHR     = pg.mkPen('c', width=1.5)
PEN_TOCO           = pg.mkPen('r', width=1.5)
PEN_FILTERED_TOCO  = pg.mkPen('r', width=1.5)
PEN_BASELINE       = pg.mkPen(color='r',       width=2, style=Qt.PenStyle.DashLine)
PEN_TRUE_PAPER_BASELINE  = pg.mkPen(color='#2ca02c', width=2, style=Qt.PenStyle.DashLine)
PEN_REMO_BASELINE  = pg.mkPen(color="#6d20a4", width=2, style=Qt.PenStyle.DashLine)
PEN_TOCO_BASELINE  = pg.mkPen(color='#555555', width=2, style=Qt.PenStyle.DashLine)
PEN_GRID_MINOR     = pg.mkPen(color='#cccccc', width=1)
PEN_GRID_MAJOR     = pg.mkPen(color='black',   width=2)
PEN_UC_BORDER      = pg.mkPen(color=(200, 0, 0), width=2)

BRUSH_ACCEL        = (0,   255, 0,   50)
BRUSH_DECEL        = (255, 0,   0,   50)
BRUSH_UC           = (255, 0,   0,   30)


class NumericMetricsDialog(QDialog):
    def __init__(self, current_base, current_var, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Window Metrics")
        
        layout = QFormLayout(self)

        # Baseline Number Input
        self.base_spin = QSpinBox()
        self.base_spin.setRange(50, 250)
        # Handle NaNs safely by defaulting to 120 if missing
        self.base_spin.setValue(int(current_base) if not np.isnan(current_base) else 120)

        # Variability Number Input
        self.var_spin = QDoubleSpinBox()
        self.var_spin.setRange(0.0, 100.0)
        self.var_spin.setDecimals(2)
        self.var_spin.setValue(float(current_var) if not np.isnan(current_var) else 0.0)

        layout.addRow("Baseline (bpm):", self.base_spin)
        layout.addRow("Variability (bpm):", self.var_spin)

        # OK / Cancel Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        return self.base_spin.value(), self.var_spin.value()


class PlotArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.win = pg.GraphicsLayoutWidget()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.win)

        self._setup_plots()
        self._setup_curves()
        self._init_item_lists()

        # Speed state
        self.current_speed_mm_per_min = 30
        self.p1.getViewBox().sigResized.connect(self._apply_speed)

    # ── Plot setup ──────────────────────────────────────────────────────────

    def _setup_plots(self):
        self.p1 = self.win.addPlot(title="Fetal Heart Rate (bpm)")
        self.p1.setLabel('left', "FHR", units='bpm')
        self.p1.setYRange(60, 200, padding=0)
        self.p1.setLimits(yMin=60, yMax=200, xMin=0)
        self.p1.getAxis('left').setTickSpacing(10, 5)
        self.p1.getAxis('bottom').setTickSpacing(60, 20)
        self.p1.showGrid(x=False, y=True)
        self.p1.addLegend()

        self.win.nextRow()

        self.p2 = self.win.addPlot(title="Uterine Contractions (TOCO)")
        self.p2.setLabel('left', "Toco")
        self.p2.setLabel('bottom', "Time", units='s')
        self.p2.setYRange(0, 100)
        self.p2.setLimits(yMin=0, yMax=100)
        self.p2.getAxis('bottom').setTickSpacing(60, 20)
        self.p2.showGrid(x=False, y=True)
        self.p2.setXLink(self.p1)


    def _setup_curves(self):
        """Create the main signal curves (no data yet)."""
        #self.curve_filled = self.p1.plot(pen=PEN_FILLED_FHR,    name="Interpolated")
        self.curve_raw    = self.p1.plot(pen=PEN_RAW_FHR,       name="Raw",          connect="finite")
        self.curve_toco   = self.p2.plot(pen=PEN_TOCO,          name="TOCO")

    def _init_item_lists(self):
        """All annotation item lists in one place."""
        self.baseline_items       = []
        self.final_baseline_items = []
        self.remo_baselines_items = []
        self.toco_baseline_items  = []
        self.status_items         = []
        self.accel_regions        = []
        self.accel_labels         = []
        self.decel_regions        = []
        self.decel_labels         = []
        self.contraction_regions  = []
        self.contraction_labels   = []
        self.fm_items = []
        self.grid_lines           = []

    # ── Data update ─────────────────────────────────────────────────────────

    def update_signals(self, time_data, fhr_clean, raw_toco):
        self.curve_raw.setData(time_data, fhr_clean)
        #self.curve_filled.setData(time_data, fhr_filled)
        self.curve_toco.setData(time_data, raw_toco)

    def clear_annotations(self):
        all_items = (
            self.baseline_items + self.final_baseline_items + self.remo_baselines_items +
            self.toco_baseline_items + self.status_items +
            self.accel_regions + self.accel_labels +
            self.decel_regions + self.decel_labels +
            self.contraction_regions + self.contraction_labels + self.fm_items +
            self.grid_lines
        )
        for item in all_items:
            for plot in [self.p1, self.p2]:
                plot.removeItem(item)

        self._init_item_lists()

    # ── Drawing ─────────────────────────────────────────────────────────────

    def draw_grid_lines(self, total_seconds):
        for t in np.arange(0, total_seconds + 60, 60):
            is_major = (t % 600 == 0)
            pen = PEN_GRID_MAJOR if is_major else PEN_GRID_MINOR
            for plot in [self.p1, self.p2]:
                line = pg.InfiniteLine(pos=t, angle=90, pen=pen)
                plot.addItem(line)
                self.grid_lines.append(line)

    def draw_toco_baselines(self, toco_segments):
        for seg in toco_segments:
            t_start, t_end = seg['time_seconds']
            val = seg['baseline']
            line = self.p2.plot([t_start, t_end], [val, val], pen=PEN_TOCO_BASELINE)
            self.toco_baseline_items.append(line)

    def draw_fhr_baselines(self, rounded_bases, true_paper_baseline, bases_class,
                            variabilities, var_class):
        window_sec = 600
        for i, val in enumerate(rounded_bases):
            if np.isnan(val):
                continue

            t_start = i * window_sec
            t_end   = t_start + window_sec

            # True Paper baseline
            f_val  = true_paper_baseline[i]
            f_line = self.p1.plot([t_start, t_end], [f_val, f_val], pen=PEN_TRUE_PAPER_BASELINE)
            self.final_baseline_items.append(f_line)

            # Window info label
            b_status = bases_class[i]
            v_status = var_class[i] if i < len(var_class) else "N/A"
            html = f"""
                <div style="background-color: rgba(255,255,255,180); padding:4px;
                            border:1px solid gray; border-radius:3px;">
                    <b style="color:black; font-size:12pt;">Window {i+1}:</b><br>
                    <span style="color:#2ca02c; font-size:11pt;">Baseline: {f_val:.0f} - {b_status}</span><br>
                    <span style="color:blue; font-size:11pt;">
                        Variability: {variabilities[i]:.2f} - {v_status}
                    </span>
                </div>
            """
            label = pg.TextItem(html=html, anchor=(0, 0))
            
            # ── INJECT METADATA & BIND LINE ──
            label.ctg_window_idx = i
            label.ctg_base_val   = f_val
            label.ctg_var_val    = variabilities[i]
            label.ctg_line_item  = f_line 
            label.ctg_t_start    = t_start
            label.ctg_t_end      = t_end   
            
            label.mouseDoubleClickEvent = lambda evnt, lbl=label: self._on_status_double_click(evnt, lbl)

            self.p1.addItem(label)
            label.setPos(t_start + 10, 195)
            self.status_items.append(label)

    def draw_contractions(self, contractions, chunk_offset_seconds=0.0, chunk_signal=None):
        """Drawing contractions (p2)."""
        for i, con in enumerate(contractions):
            abs_start = con['start_seconds'] + chunk_offset_seconds
            abs_end = con['end_seconds'] + chunk_offset_seconds

            region_p2 = pg.LinearRegionItem(
                values=[abs_start, abs_end],
                movable=True, brush=BRUSH_UC, pen=PEN_UC_BORDER
            )
            self.p2.addItem(region_p2)
            self.contraction_regions.append(region_p2)

            # Inject metadata and the signal array
            region_p2.ctg_meta = con
            region_p2.ctg_fs = 4.0
            region_p2.ctg_offset = chunk_offset_seconds
            region_p2.ctg_signal = chunk_signal  # <--- Store the signal here!
            
            region_p2.sigRegionChangeFinished.connect(self._on_contraction_resized)
            region_p2.mouseDoubleClickEvent = lambda ev, reg=region_p2: self._on_contraction_double_click(ev, reg)
            
            mid = (abs_start + abs_end) / 2
            html = f"""
                <div style="background-color: rgba(255,255,255,180); padding:2px;
                            border:1px solid gray; border-radius:3px;">
                    <span style="color:black; font-weight:bold; font-size:10pt;">UC</span>
                </div>
            """
            label = pg.TextItem(html=html, anchor=(0.5, 0))
            label.ctg_meta = con  
            label.setPos(mid, 90)
            self.p2.addItem(label)
            self.contraction_labels.append(label)

            region_p2.linked_label = label
            region_p2.sigRegionChanged.connect(self._on_region_dragging)
            label.mouseDoubleClickEvent = lambda evnt, lbl=label: self._on_label_double_click(evnt, lbl)

    def _on_contraction_resized(self, region):
        """Call when contraction is resized."""
        abs_start_sec, abs_end_sec = region.getRegion()
        fs = getattr(region, 'ctg_fs', 4.0)
        offset = getattr(region, 'ctg_offset', 0.0)

        rel_start = abs_start_sec - offset
        rel_end   = abs_end_sec   - offset

        start_idx = int(max(0, round(rel_start * fs)))
        end_idx   = int(round(rel_end * fs))

        region.ctg_meta['start_seconds'] = float(max(0.0, rel_start))
        region.ctg_meta['end_seconds']   = float(rel_end)
        region.ctg_meta['start_idx']     = start_idx
        region.ctg_meta['end_idx']       = end_idx
        region.ctg_meta['duration']      = float(rel_end - rel_start)

        # ── Find the true peak dynamically upon drag release ──
        if hasattr(region, 'ctg_signal') and region.ctg_signal is not None:
            safe_end = min(len(region.ctg_signal), end_idx)
            segment = region.ctg_signal[start_idx:safe_end]
            
            if len(segment) > 0:
                region.ctg_meta['peak_s'] = float((start_idx + np.argmax(segment)) / fs)
            else:
                region.ctg_meta['peak_s'] = float((rel_start + rel_end) / 2)
        else:
            region.ctg_meta['peak_s'] = float((rel_start + rel_end) / 2)

        print(f"Modificado via arrastar: start={rel_start:.2f}s  end={rel_end:.2f}s  peak_s={region.ctg_meta['peak_s']:.2f}s")

    def _on_status_double_click(self, ev, label):
        """Opens dialog box and changes baseline line."""
        ev.accept()
        
        dialog = NumericMetricsDialog(label.ctg_base_val, label.ctg_var_val, self)
        
        if dialog.exec():
            new_base_val, new_var_val = dialog.get_values()
            
            # Request re-classification from DataLoader
            if hasattr(self, '_callback_update_status'):
                new_base_class, new_var_class = self._callback_update_status(
                    label.ctg_window_idx, new_base_val, new_var_val
                )
            else:
                new_base_class, new_var_class = "N/A", "N/A"
            
            # Update internal label metadata
            label.ctg_base_val = new_base_val
            label.ctg_var_val = new_var_val
            
            # Move the dashed line on the plot!
            label.ctg_line_item.setData(
                [label.ctg_t_start, label.ctg_t_end], 
                [new_base_val, new_base_val]
            )

            # Rebuild and apply the HTML with the new numbers and auto-classifications
            html = f"""
                <div style="background-color: rgba(255,255,255,180); padding:4px;
                            border:1px solid gray; border-radius:3px;">
                    <b style="color:black; font-size:12pt;">Window {label.ctg_window_idx+1}:</b><br>
                    <span style="color:#2ca02c; font-size:11pt;">Baseline: {new_base_val:.0f} - {new_base_class}</span><br>
                    <span style="color:blue; font-size:11pt;">
                        Variability: {new_var_val:.2f} - {new_var_class}
                    </span>
                </div>
            """
            label.setHtml(html)


    def _on_contraction_double_click(self, ev, region):
        """Remove a contração ao fazer duplo clique na região."""
        ev.accept()
        meta = region.ctg_meta

        # Remove region on p2
        try:
            self.p2.removeItem(region)
        except Exception:
            pass
            
        if region in self.contraction_regions:
            self.contraction_regions.remove(region)

        # Remove UC label
        for label in list(self.contraction_labels):
            if getattr(label, 'ctg_meta', None) is meta:
                self.p2.removeItem(label)
                self.contraction_labels.remove(label)

        # Notify dataloader to remove it from memory
        if hasattr(self, '_callback_delete_contraction'):
            self._callback_delete_contraction(meta)

        print(f"Contraction removed: start={meta.get('start_seconds'):.2f}s")

    def draw_fetal_movs(self, fetal_movs):
        if fetal_movs is None or len(fetal_movs) == 0:
            return

        y_range     = self.p2.getViewBox().viewRange()[1]
        y_top       = y_range[1]
        tick_height = (y_range[1] - y_range[0]) * 0.05

        x_data, y_data = [], []
        for fm in fetal_movs:
            fm_time = fm / 4  # sample index → seconds
            x_data += [fm_time, fm_time]
            y_data += [y_top - tick_height, y_top]

        fm_item = pg.PlotDataItem(
            x=x_data,
            y=y_data,
            pen=pg.mkPen(color='black', width=2),
            connect='pairs'
        )
        self.p2.addItem(fm_item)
        self.fm_items.append(fm_item)


    def registrar_callback_clique(self, callback_funcao, callback_delete=None, cb_fhr_add=None, cb_fhr_del=None, cb_status_update=None):
        """Permite ao DataLoader registrar funções para clique (novo) e delete."""
        self._callback_novo_clique = callback_funcao
        
        if callback_delete:
            self._callback_delete_contracao = callback_delete
            
        if cb_fhr_add:
            self._callback_novo_fhr = cb_fhr_add
            
        if cb_fhr_del:
            self._callback_delete_fhr = cb_fhr_del
            
        if cb_status_update:
            self._callback_update_status = cb_status_update

        if not hasattr(self, '_click_connected'):
            self.p1.scene().sigMouseClicked.connect(self._on_plot_clicked)
            self._click_connected = True


    def _on_plot_clicked(self, event):
        """Detecta o clique do mouse no gráfico p1 ou p2."""
        modifiers = QApplication.keyboardModifiers()

        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            pos = event.scenePos()
            
            # Clicou no FHR (p1)
            if self.p1.sceneBoundingRect().contains(pos):
                if hasattr(self, '_callback_novo_fhr'):
                    mouse_x = self.p1.getViewBox().mapSceneToView(pos).x()
                    
                    if event.button() == Qt.MouseButton.RightButton:
                        self._callback_novo_fhr(mouse_x, "acceleration")
                        event.accept()
                    elif event.button() == Qt.MouseButton.LeftButton:
                        self._callback_novo_fhr(mouse_x, "deceleration")
                        event.accept()
            
            # Clicou no TOCO (p2)
            elif self.p2.sceneBoundingRect().contains(pos):
                if hasattr(self, '_callback_novo_clique'):
                    if event.button() == Qt.MouseButton.RightButton:
                        mouse_x = self.p2.getViewBox().mapSceneToView(pos).x()
                        self._callback_novo_clique(mouse_x)
                        event.accept()


    def _on_region_dragging(self, region):
        """Updates the label position in real-time while the region is dragged."""
        if hasattr(region, 'linked_label'):
            minX, maxX = region.getRegion()
            midX = (minX + maxX) / 2
            
            # Keep the label's original Y position, just move X
            current_y = region.linked_label.pos().y()
            region.linked_label.setPos(midX, current_y)

    def _on_label_double_click(self, ev, label):
        """Opens a dropdown dialog to select predefined labels."""
        ev.accept()
        meta = getattr(label, 'ctg_meta', {})
        
        # Check what area we are in (FHR vs TOCO)
        e_type = meta.get('type')
        current_text = meta.get('sub-type', 'UC')
        
        if e_type == 'acceleration':
            options = ["Acceleration", "Prolonged Acceleration", "Shoulder"]
        elif e_type == 'deceleration': #set another round of classification??
            options = ["Early Deceleration", "Late Deceleration", "Variable Typical", "Variable Atypical", "Moderate Deceleration", "Severe Deceleration"]
        else: # TOCO (Uterine Contractions)
            options = ["UC"]
            
        try:
            default_index = options.index(current_text)
        except ValueError:
            options.insert(0, current_text)
            default_index = 0

        new_text, ok = QInputDialog.getItem(self, "Select Label", "Choose the event type:", options, default_index, False)
        
        if ok and new_text:
            meta['sub-type'] = new_text
            
            if e_type == 'acceleration':
                color, font_size, padding = 'green', '13pt', '4px'
            elif e_type == 'deceleration':
                color, font_size, padding = 'red', '13pt', '4px'
            else:
                color, font_size, padding = 'black', '13pt', '2px'
            
            html = f"""
                <div style="background-color: rgba(255,255,255,180); padding:{padding};
                            border:1px solid gray; border-radius:3px;">
                    <span style="color:{color}; font-weight:bold; font-size:{font_size};">{new_text}</span>
                </div>
            """
            label.setHtml(html)




    def draw_events(self, class_events):
        for ev in class_events:
            e_type = ev.get('type')
            e_subtype = ev.get('sub-type')

            if e_subtype != False:
                if e_type == 'acceleration':
                    brush      = BRUSH_ACCEL
                    text_color = 'green'
                    y_pos      = 195
                elif e_type == 'deceleration':
                    brush      = BRUSH_DECEL
                    text_color = 'red'
                    y_pos      = 160
                else:
                    continue
            else:
                continue

            region = pg.LinearRegionItem(
                values=[ev['start_seconds'], ev['end_seconds']],
                movable=True, brush=brush, pen=pg.mkPen(None)
            )
            
            # INJECT METADATA & BIND EVENTS
            region.ctg_meta = ev
            region.sigRegionChangeFinished.connect(self._on_event_resized)
            region.mouseDoubleClickEvent = lambda evnt, reg=region: self._on_event_double_click(evnt, reg)
            
            self.p1.addItem(region)

            html = f"""
                <div style="background-color: rgba(255,255,255,180); padding:4px;
                            border:1px solid gray; border-radius:3px;">
                    <span style="color:{text_color}; font-size:14pt;">{ev['sub-type']}</span>
                </div>
            """
            label = pg.TextItem(html=html, anchor=(0.5, 0))
            region.linked_label = label
            region.sigRegionChanged.connect(self._on_region_dragging)
            label.mouseDoubleClickEvent = lambda evnt, lbl=label: self._on_label_double_click(evnt, lbl)
            label.ctg_meta = ev # Bind for deletion
            label.setPos((ev['start_seconds'] + ev['end_seconds']) / 2, y_pos)
            self.p1.addItem(label)

            if e_type == 'acceleration':
                self.accel_regions.append(region)
                self.accel_labels.append(label)
            else:
                self.decel_regions.append(region)
                self.decel_labels.append(label)

    # ── Add these two new helper methods for FHR events ─────────────────────

    def _on_event_resized(self, region):
        """Updates dictionary when an FHR event edge is dragged."""
        abs_start_sec, abs_end_sec = region.getRegion()
        
        region.ctg_meta['start_seconds'] = float(max(0.0, abs_start_sec))
        region.ctg_meta['end_seconds']   = float(abs_end_sec)
        region.ctg_meta['start_idx']     = int(max(0, round(abs_start_sec * 4.0)))
        region.ctg_meta['end_idx']       = int(round(abs_end_sec * 4.0))

        print(f"FHR event modified: start={abs_start_sec:.2f}s end={abs_end_sec:.2f}s")

    def _on_event_double_click(self, ev, region):
        """Removes an FHR event on double-click."""
        ev.accept()
        meta = region.ctg_meta

        # Remove visuals from p1
        try:
            self.p1.removeItem(region)
        except Exception:
            pass
            
        if region in self.accel_regions: self.accel_regions.remove(region)
        if region in self.decel_regions: self.decel_regions.remove(region)

        # Remove the associated text label
        for label_list in [self.accel_labels, self.decel_labels]:
            for label in list(label_list):
                if getattr(label, 'ctg_meta', None) is meta:
                    self.p1.removeItem(label)
                    label_list.remove(label)

        # Notify DataLoader
        if hasattr(self, '_callback_delete_fhr'):
            self._callback_delete_fhr(meta)
                
    # ── Speed ────────────────────────────────────────────────────────────────

    def change_speed(self, mm_per_min):
        self.current_speed_mm_per_min = mm_per_min
        tick_map = {10: (60, 60), 20: (60, 30), 30: (60, 20)}
        major, minor = tick_map.get(mm_per_min, (60, 20))
        for plot in [self.p1, self.p2]:
            plot.getAxis('bottom').setTickSpacing(major, minor)
        self.p1.setYRange(60, 200, padding=0)
        self.p2.setYRange(0, 100)
        self._apply_speed()

    def _apply_speed(self):
        screen = QApplication.primaryScreen()
        dpi = screen.physicalDotsPerInchX()
        mm_per_pixel = 25.4 / dpi

        plot_width_px = self.p1.getViewBox().width()
        if plot_width_px <= 0:
            return

        plot_width_mm   = plot_width_px * mm_per_pixel
        mm_per_sec      = self.current_speed_mm_per_min / 60.0
        visible_seconds = plot_width_mm / mm_per_sec

        x_start = max(0, self.p1.viewRange()[0][0])
        self.p1.setXRange(x_start, x_start + visible_seconds, padding=0)

    def reset_view(self):
        self.p1.setXRange(0, 1, padding=0)
        self._apply_speed()
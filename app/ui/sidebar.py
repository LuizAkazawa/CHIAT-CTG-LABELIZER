# app/ui/sidebar.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox,
                              QCheckBox, QPushButton, QButtonGroup, QListWidget)
from PyQt6.QtCore import Qt, pyqtSignal
import os

SIDEBAR_STYLE = """
    QWidget {
        background-color: #f5f5f5;
        color: black;
    }
    QGroupBox {
        color: black;
        border: 1px solid #ccc;
        margin-top: 8px;
        font-weight: 500;
    }
    QGroupBox::title {
        color: black;
        subcontrol-origin: margin;
        left: 8px;
    }
    QCheckBox {
        color: black;
    }
    QCheckBox::indicator {
        width: 13px;
        height: 13px;
        border: 1.5px solid black;
        border-radius: 2px;
        background-color: white;
    }
    QCheckBox::indicator:checked {
        background-color: #378ADD;
        border: 1.5px solid black;
    }
    QPushButton {
        color: black;
        border: 1px solid #aaa;
        padding: 5px;
        background-color: white;
    }
    QPushButton:hover {
        background-color: #e0e0e0;
    }
"""

class Sidebar(QWidget):

    # --- Signals emitted to the outside world ---
    load_requested = pyqtSignal()
    save_requested = pyqtSignal()  
    speed_changed  = pyqtSignal(int)   # emits mm/min: 10, 20, or 30
    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(180)
        self.setStyleSheet(SIDEBAR_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(14)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Build remaining UI pieces
        layout.addWidget(self._build_speed_group())
        layout.addWidget(self._build_file_browser())
        layout.addStretch()
        layout.addWidget(self._build_action_buttons())


    # ── Group builders ──────────────────────────────────────────────

    def _build_file_browser(self):
        group, inner = self._make_group("Dataset Files")
        
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        
        # Scan the 'Data' folder for .cts files
        data_dir = "Data"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith(".cts")]
            for f in sorted(files):
                self.file_list.addItem(f)
        else:
            self.file_list.addItem("Data folder not found")
            self.file_list.setEnabled(False)

        # Connect double-click to our custom signal
        self.file_list.itemDoubleClicked.connect(self._on_file_double_clicked)
        
        inner.addWidget(self.file_list)
        return group

    def _on_file_double_clicked(self, item):
        filename = item.text()
        filepath = os.path.join("Data", filename)
        self.file_selected.emit(filepath)


    def _build_speed_group(self):
        group, inner = self._make_group("Speed")

        self.cb_1cm = QCheckBox("1 cm/min")
        self.cb_2cm = QCheckBox("2 cm/min")
        self.cb_3cm = QCheckBox("3 cm/min")
        self.cb_3cm.setChecked(True)

        self._speed_map = {
            self.cb_1cm: 10,
            self.cb_2cm: 20,
            self.cb_3cm: 30,
        }

        self.speed_btn_group = QButtonGroup()
        self.speed_btn_group.setExclusive(True)
        for cb in (self.cb_1cm, self.cb_2cm, self.cb_3cm):
            self.speed_btn_group.addButton(cb)
            inner.addWidget(cb)

        self.speed_btn_group.buttonClicked.connect(
            lambda btn: self.speed_changed.emit(self._speed_map[btn])
        )
        return group
    
    def _build_action_buttons(self):
        """Agrupa os botões de carregar e salvar na parte inferior da barra."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.btn_load = QPushButton("Load CTS File")
        self.btn_load.clicked.connect(self.load_requested)

        # Novo botão de Salvar
        self.btn_save = QPushButton("Save changes")
        self.btn_save.setStyleSheet(
            "font-weight: bold; background-color: #e3f2fd; border-color: #2196f3;"
        )
        self.btn_save.clicked.connect(self.save_requested)

        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_save)
        return container

    # ── Helpers ─────────────────────────────────────────────────────

    def _make_group(self, title):
        """Creates a QGroupBox and returns (group, inner_layout)."""
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        return group, layout
import sys
from PyQt6.QtWidgets import QApplication
from app.viewer import CTGInteractiveViewer

app = QApplication(sys.argv)
viewer = CTGInteractiveViewer()
viewer.show()
sys.exit(app.exec())

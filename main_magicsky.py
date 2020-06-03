import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QTableWidgetItem
from PyQt5.QtCore import QEventLoop, QTimer
from Magic_Sky import Ui_MainWindow

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import numpy as np

class Figure_Canva(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure()
        super(Figure_Canva).__init__(self.fig)
        self.setParent(parent)
        self.axes = plt.subplot(111)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
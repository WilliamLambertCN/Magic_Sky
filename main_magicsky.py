import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QTableWidgetItem
from PyQt5.QtCore import QEventLoop, QTimer
from Magic_Sky import Ui_MainWindow

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import numpy as np

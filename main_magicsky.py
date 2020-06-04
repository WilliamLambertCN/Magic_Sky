import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QEventLoop, QTimer
from Magic_Sky import Ui_MainWindow
import cv2
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

class EmittingStr(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))
        loop = QEventLoop()
        QTimer.singleShot(100, loop.quit)
        loop.exec_()

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)
        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)
        self.pushButton.clicked.connect(self.load_source)
        self.pushButton_2.clicked.connect(self.load_target)
        self.pushButton_3.clicked.connect(self.show_input)
        self.pushButton_4.clicked.connect(self.show_result)
        self.pushButton_5.clicked.connect(self.save_result)
        self.pushButton_skyrpl.clicked.connect(self.skyrpl)

    def outputWritten(self, text):
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def load_source(self):
        print('Loading source file')
        if self.modetext.currentText() == 'Photo':
            file_filter = r"IMAGE(*.jpg;*.jpeg;*.png);;ALL FILE(*)"
        else:
            file_filter = r"VIDEO(*.mp4);;ALL FILE(*)"
        self.srcname, file_type = QFileDialog.getOpenFileName(self, caption='source', directory='Demo',
                                                              filter=file_filter)
        if self.srcname != "":
            print('Load source successfully: {0}'.format(self.srcname))
            self.sourcetext.setText(self.srcname)
            print('Loaded source file')
        else:
            print('Load failed.')
        if self.modetext.currentText() == 'Photo' and (self.srcname != ''):
            self.src = cv2.imread(self.srcname)
            self.src = cv2.cvtColor(self.src, cv2.COLOR_BGR2RGB)
            self.src_h, self.src_w, self.src_c = self.src.shape

    def load_target(self):
        print('Loading target file')
        file_filter = r"IMAGE(*.jpg;*.jpeg;*.png);;ALL FILE(*)"
        self.tgtname, file_type = QFileDialog.getOpenFileName(self, caption='target', directory='sky',
                                                              filter=file_filter)
        if self.tgtname != "":
            print('Load target successfully: {0}'.format(self.tgtname))
            self.targettext.setText(self.tgtname)
            print('Loaded target file')
        else:
            print('Load failed.')
        if self.tgtname != '':
            self.tgt = cv2.imread(self.tgtname)
            self.tgt = cv2.cvtColor(self.tgt, cv2.COLOR_BGR2RGB)
            self.tgt_h, self.tgt_w, self.tgt_c = self.src.shape

    def scene_show(self, filename, graphic_view):
        assert isinstance(graphic_view, QtWidgets.QGraphicsView)
        h, w, c = cv2.imread(filename).shape
        frame = QtGui.QImage(filename)
        pix = QtGui.QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scence = QGraphicsScene()
        scale = min(graphic_view.height() / (1.02 * h),
                    graphic_view.width() / (1.02 * w))
        item.setScale(scale)
        scence.addItem(item)
        graphic_view.setScene(scence)
        return

    def skyrpl(self):
        print('Replacing Sky')
        print('Sky Replaced')

    def show_input(self):
        print('Show input')
        self.scene_show(self.srcname, self.sourceView)
        self.scene_show(self.tgtname, self.targetView)

    def show_result(self):
        print('Show result')

    def save_result(self):
        print('Saving result.')
        print("Saved result.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import QVideoWidget

from Magic_Sky import Ui_MainWindow
import cv2
from func import video_replace, photo_replace
import torch
from tools.common_tools import set_seed
from tools.unet import UNet

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

        self.checkpoint_load = 'tools/checkpoint_199_epoch.pkl'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = UNet(in_channels=3, out_channels=1, init_features=32)  # init_features is 64 in stander uent
        self.net.to(self.device)
        if self.checkpoint_load is not None:
            checkpoint = torch.load(self.checkpoint_load)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            print('load checkpoint from %s' % self.checkpoint_load)
        else:
            raise Exception("\nPlease specify the checkpoint")
        set_seed()  # 设置随机种子

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
        if self.modetext.currentText() == 'Photo':
            result = photo_replace(self.src, self.tgt, self.net)
            cv2.imwrite('temp/results.jpg', result[:, :, ::-1])
            self.scene_show('temp/results.jpg', self.resultView)
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

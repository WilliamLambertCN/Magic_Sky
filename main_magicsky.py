import os
import shutil
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QEventLoop, QTimer, QUrl, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage, QIcon
# from PyQt5.QtMultimedia import *
# from PyQt5.QtMultimediaWidgets import QVideoWidget
import time
from Magic_Sky import Ui_MainWindow
import cv2
from func import video_replace, photo_replace, photo_infer, photo_improve
import torch
from tools.common_tools import set_seed
from tools.unet import UNet

class EmittingStr(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)  # emit str signal

    def write(self, text):
        self.textWritten.emit(str(text))
        loop = QEventLoop()
        QTimer.singleShot(100, loop.quit)
        loop.exec_()

class Thread(QThread):

    def __init__(self):
        super(Thread, self).__init__()

    def run(self, func, arg):
        func(*arg)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(QIcon('Software GUI/beauty.ico'))
        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)
        self.pushButton.clicked.connect(self.load_source)
        self.pushButton_2.clicked.connect(self.load_target)
        self.pushButton_3.clicked.connect(self.play_input_video)
        self.pushButton_4.clicked.connect(self.play_result_video)
        self.pushButton_5.clicked.connect(self.save_result)
        self.pushButton_skyrpl.clicked.connect(self.skyrpl)

        # self.checkpoint_load = 'test6_lovasz_1e-2/checkpoint_19_epoch.pkl'
        # self.checkpoint_load = 'test6_lovasz_1e-2/bestdice_min_38.57%_checkpoint_55_epoch.pkl'
        # self.checkpoint_load = 'test4_lovasz_1e-2/bestdice_min_47.90%_checkpoint_35_epoch.pkl'
        self.checkpoint_load = 'tools/checkpoint_199_epoch.pkl'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = UNet(in_channels=3, out_channels=1, init_features=32)  # init_features is 64 in stander uent
        self.net.to(self.device)
        self.net.eval()
        if self.checkpoint_load is not None:
            checkpoint = torch.load(self.checkpoint_load)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            print(
                    '\nWelcome to use Magic Sky Software. \nPytorch model loads checkpoint from %s' % self.checkpoint_load)
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
            self.scene_show(self.srcname, self.sourceView)
        elif self.modetext.currentText() == 'Video' and (self.srcname != ''):
            cap = cv2.VideoCapture(self.srcname)
            success, frame = cap.read()
            assert success
            cv2.imwrite('temp/frame1.jpg', frame)
            cap.release()
            self.scene_show('temp/frame1.jpg', self.sourceView)

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
            self.tgt_h, self.tgt_w, self.tgt_c = self.tgt.shape
            self.scene_show(self.tgtname, self.targetView)

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
        elif self.modetext.currentText() == 'Video':
            self.cap = cv2.VideoCapture(self.srcname)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frameCount = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.videoWriter = cv2.VideoWriter('temp/result.mp4',
                                               cv2.VideoWriter_fourcc(*'mp4v'),
                                               fps, size)
            assert self.cap.isOpened()

            tic = time.time()
            # def video_thread(frameCount, tic):
            for index in range(int(frameCount)):
                success, frame = self.cap.read()
                if success:
                    result = photo_replace(frame[..., ::-1], self.tgt, self.net, 1)
                    if index == 0:
                        cv2.imwrite('temp/frame1.jpg', frame)
                        cv2.imwrite('temp/result1.jpg', result[..., ::-1])
                        self.scene_show('temp/frame1.jpg', self.sourceView)
                        self.scene_show('temp/result1.jpg', self.resultView)
                    if index % 50 == 0:
                        print("Replace %d, time %.2f" % (index, (time.time() - tic)))
                    self.videoWriter.write(result[..., ::-1])
                else:
                    assert success
            self.cap.release()
            self.videoWriter.release()
            print("Infer Done! Time %.2f" % (time.time() - tic))

    def play_input_video(self):
        assert self.modetext.currentText() == 'Video'
        print('Play input video')
        global playmode
        playmode = "Input"
        global videoName  # 在这里设置全局变量以便在线程中使用
        videoName = self.srcname
        # cap = cv2.VideoCapture(str(videoName))
        self.th = Thread(self)
        self.th.changeSrcPixmap.connect(self.setInputImage)
        self.th.start()

    def play_result_video(self):
        assert self.modetext.currentText() == 'Video'
        print('Play result video')
        global playmode
        playmode = "Result"
        global videoName  # 在这里设置全局变量以便在线程中使用
        videoName = "temp/result.mp4"
        # cap = cv2.VideoCapture(str(videoName))
        self.th = Thread(self)
        self.th.changeResPixmap.connect(self.setResImage)
        self.th.start()

    def setResImage(self, Qframe):
        pix = QtGui.QPixmap.fromImage(Qframe)
        item = QGraphicsPixmapItem(pix)
        scence = QGraphicsScene()
        scale = min(self.resultView.height() / (1.02 * pix.height()),
                    self.resultView.width() / (1.02 * pix.width()))
        item.setScale(scale)
        scence.addItem(item)
        self.resultView.setScene(scence)

    def setInputImage(self, Qframe):
        pix = QtGui.QPixmap.fromImage(Qframe)
        item = QGraphicsPixmapItem(pix)
        scence = QGraphicsScene()
        scale = min(self.sourceView.height() / (1.02 * pix.height()),
                    self.sourceView.width() / (1.02 * pix.width()))
        item.setScale(scale)
        scence.addItem(item)
        self.sourceView.setScene(scence)

    def save_result(self):
        print('Saving result.')
        if self.modetext.currentText() == 'Photo':
            file_filter = r"(*.jpg)"
        else:
            file_filter = r"(*.mp4)"
        save_filename, filetype = QFileDialog.getSaveFileName(self, caption="Save result to: ", directory='results',
                                                              filter=file_filter)
        print(save_filename, ' ', filetype, ' ')
        if self.modetext.currentText() == 'Photo':
            self.mycopyfile("temp/results.jpg", save_filename)
        else:
            self.mycopyfile("temp/result.mp4", save_filename)

    def mycopyfile(self, srcfile, dstfile):
        assert srcfile.endswith(('mp4', 'jpg'))
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        print("Results saved -> %s" % (dstfile))

class Thread(QThread):  # 采用线程来播放视频

    changeResPixmap = pyqtSignal(QtGui.QImage)
    changeSrcPixmap = pyqtSignal(QtGui.QImage)

    def run(self):
        assert playmode in ["Input", "Result"]
        cap = cv2.VideoCapture(videoName)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(videoName)
        while (cap.isOpened() == True):
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                                 QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
                # p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                if playmode == "Input":
                    self.changeSrcPixmap.emit(convertToQtFormat)
                elif playmode == "Result":
                    self.changeResPixmap.emit(convertToQtFormat)
                time.sleep(1 / (1.1 * fps))  # 控制视频播放的速度
            else:
                break

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

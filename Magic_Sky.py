# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Magic_Sky.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1163, 855)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(540, 40, 601, 801))
        self.tabWidget.setObjectName("tabWidget")
        self.tap = QtWidgets.QWidget()
        self.tap.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tap.setObjectName("tap")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.tap)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 20, 561, 311))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_4 = QtWidgets.QGroupBox(self.horizontalLayoutWidget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.sourceView = QtWidgets.QGraphicsView(self.groupBox_4)
        self.sourceView.setGeometry(QtCore.QRect(11, 20, 261, 281))
        self.sourceView.setObjectName("sourceView")
        self.horizontalLayout.addWidget(self.groupBox_4)
        self.groupBox_5 = QtWidgets.QGroupBox(self.horizontalLayoutWidget)
        self.groupBox_5.setObjectName("groupBox_5")
        self.targetView = QtWidgets.QGraphicsView(self.groupBox_5)
        self.targetView.setGeometry(QtCore.QRect(10, 20, 261, 281))
        self.targetView.setObjectName("targetView")
        self.horizontalLayout.addWidget(self.groupBox_5)
        self.groupBox_6 = QtWidgets.QGroupBox(self.tap)
        self.groupBox_6.setGeometry(QtCore.QRect(20, 340, 561, 421))
        self.groupBox_6.setObjectName("groupBox_6")
        self.resultView = QtWidgets.QGraphicsView(self.groupBox_6)
        self.resultView.setGeometry(QtCore.QRect(12, 20, 541, 391))
        self.resultView.setObjectName("resultView")
        self.tabWidget.addTab(self.tap, "")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(30, 90, 481, 751))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 451, 731))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.groupBox.setObjectName("groupBox")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 360, 411, 361))
        self.groupBox_3.setObjectName("groupBox_3")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox_3)
        self.textBrowser.setGeometry(QtCore.QRect(10, 20, 391, 331))
        self.textBrowser.setObjectName("textBrowser")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 190, 411, 131))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 0, 1, 1, 1)
        self.pushButton_skyrpl = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_skyrpl.setObjectName("pushButton_skyrpl")
        self.gridLayout.addWidget(self.pushButton_skyrpl, 0, 2, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 1, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 1, 1, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout.addWidget(self.pushButton_5, 1, 2, 1, 1)
        self.formLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 50, 411, 111))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.modetilte = QtWidgets.QLabel(self.formLayoutWidget)
        self.modetilte.setFocusPolicy(QtCore.Qt.NoFocus)
        self.modetilte.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.modetilte.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.modetilte.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.modetilte.setObjectName("modetilte")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.modetilte)
        self.line = QtWidgets.QFrame(self.formLayoutWidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.line)
        self.line_2 = QtWidgets.QFrame(self.formLayoutWidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.line_2)
        self.sourcetitle = QtWidgets.QLabel(self.formLayoutWidget)
        self.sourcetitle.setObjectName("sourcetitle")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.sourcetitle)
        self.sourcetext = QtWidgets.QLabel(self.formLayoutWidget)
        self.sourcetext.setObjectName("sourcetext")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.sourcetext)
        self.targettitle = QtWidgets.QLabel(self.formLayoutWidget)
        self.targettitle.setObjectName("targettitle")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.targettitle)
        self.targettext = QtWidgets.QLabel(self.formLayoutWidget)
        self.targettext.setObjectName("targettext")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.targettext)
        self.modetext = QtWidgets.QComboBox(self.formLayoutWidget)
        self.modetext.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLength)
        self.modetext.setObjectName("modetext")
        self.modetext.addItem("")
        self.modetext.addItem("")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.modetext)
        self.verticalLayout.addWidget(self.groupBox)
        self.Slogan = QtWidgets.QLabel(self.centralwidget)
        self.Slogan.setGeometry(QtCore.QRect(40, 10, 471, 71))
        font = QtGui.QFont()
        font.setFamily("Monotype Corsiva")
        font.setPointSize(26)
        font.setItalic(True)
        self.Slogan.setFont(font)
        self.Slogan.setAlignment(QtCore.Qt.AlignCenter)
        self.Slogan.setObjectName("Slogan")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSource_Image = QtWidgets.QAction(MainWindow)
        self.actionSource_Image.setObjectName("actionSource_Image")
        self.actionTarget_Sky = QtWidgets.QAction(MainWindow)
        self.actionTarget_Sky.setObjectName("actionTarget_Sky")
        self.actionLoad_Source_video = QtWidgets.QAction(MainWindow)
        self.actionLoad_Source_video.setObjectName("actionLoad_Source_video")
        self.actionLoad_Target_Sky = QtWidgets.QAction(MainWindow)
        self.actionLoad_Target_Sky.setObjectName("actionLoad_Target_Sky")

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Magic Sky (V1.1)"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Source"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Target"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Result"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tap), _translate("MainWindow", "Display"))
        self.groupBox.setTitle(_translate("MainWindow", "Setting"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Console"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Command"))
        self.pushButton.setText(_translate("MainWindow", "Load Source"))
        self.pushButton_2.setText(_translate("MainWindow", "Load Target"))
        self.pushButton_skyrpl.setText(_translate("MainWindow", "Magic Sky!"))
        self.pushButton_3.setText(_translate("MainWindow", "Show Input"))
        self.pushButton_4.setText(_translate("MainWindow", "Show Result"))
        self.pushButton_5.setText(_translate("MainWindow", "Save Result"))
        self.modetilte.setText(_translate("MainWindow", "     Mode:"))
        self.sourcetitle.setText(_translate("MainWindow", "   Source:"))
        self.sourcetext.setText(_translate("MainWindow", "None"))
        self.targettitle.setText(_translate("MainWindow", "  Tartget:"))
        self.targettext.setText(_translate("MainWindow", "None"))
        self.modetext.setItemText(0, _translate("MainWindow", "Photo"))
        self.modetext.setItemText(1, _translate("MainWindow", "Video"))
        self.Slogan.setText(_translate("MainWindow", "The sky full of magic"))
        self.actionSource_Image.setText(_translate("MainWindow", "Source Picture"))
        self.actionTarget_Sky.setText(_translate("MainWindow", "Target Sky"))
        self.actionLoad_Source_video.setText(_translate("MainWindow", "Source video"))
        self.actionLoad_Target_Sky.setText(_translate("MainWindow", "Target Sky"))



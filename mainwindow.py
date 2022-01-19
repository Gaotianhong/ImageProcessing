# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(986, 693)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.word1 = QtWidgets.QLabel(self.centralwidget)
        self.word1.setGeometry(QtCore.QRect(0, 20, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Zapfino")
        font.setPointSize(13)
        font.setItalic(True)
        self.word1.setFont(font)
        self.word1.setAutoFillBackground(False)
        self.word1.setObjectName("word1")
        self.word2 = QtWidgets.QLabel(self.centralwidget)
        self.word2.setGeometry(QtCore.QRect(40, 50, 121, 81))
        font = QtGui.QFont()
        font.setFamily("Zapfino")
        font.setPointSize(13)
        font.setItalic(True)
        self.word2.setFont(font)
        self.word2.setObjectName("word2")
        self.imageView = ImageView(self.centralwidget)
        self.imageView.setGeometry(QtCore.QRect(200, 10, 780, 650))
        self.imageView.setAutoFillBackground(True)
        self.imageView.setStyleSheet("QGraphicsView {background:transparent}")
        self.imageView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.imageView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.imageView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.imageView.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.imageView.setViewportUpdateMode(QtWidgets.QGraphicsView.MinimalViewportUpdate)
        self.imageView.setObjectName("imageView")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 130, 140, 96))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.viewOriginal = QtWidgets.QPushButton(self.layoutWidget)
        self.viewOriginal.setStyleSheet("")
        self.viewOriginal.setObjectName("viewOriginal")
        self.verticalLayout.addWidget(self.viewOriginal)
        self.drawHistogram = QtWidgets.QPushButton(self.layoutWidget)
        self.drawHistogram.setObjectName("drawHistogram")
        self.verticalLayout.addWidget(self.drawHistogram)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 240, 182, 101))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.brightness = QtWidgets.QLabel(self.layoutWidget1)
        self.brightness.setAlignment(QtCore.Qt.AlignCenter)
        self.brightness.setObjectName("brightness")
        self.verticalLayout_2.addWidget(self.brightness)
        self.Cbrightness = QtWidgets.QSlider(self.layoutWidget1)
        self.Cbrightness.setMinimum(-255)
        self.Cbrightness.setMaximum(255)
        self.Cbrightness.setProperty("value", 0)
        self.Cbrightness.setOrientation(QtCore.Qt.Horizontal)
        self.Cbrightness.setObjectName("Cbrightness")
        self.verticalLayout_2.addWidget(self.Cbrightness)
        self.contrast = QtWidgets.QLabel(self.layoutWidget1)
        self.contrast.setAlignment(QtCore.Qt.AlignCenter)
        self.contrast.setObjectName("contrast")
        self.verticalLayout_2.addWidget(self.contrast)
        self.Ccontrast = QtWidgets.QSlider(self.layoutWidget1)
        self.Ccontrast.setMinimum(-127)
        self.Ccontrast.setMaximum(127)
        self.Ccontrast.setProperty("value", 0)
        self.Ccontrast.setOrientation(QtCore.Qt.Horizontal)
        self.Ccontrast.setObjectName("Ccontrast")
        self.verticalLayout_2.addWidget(self.Ccontrast)
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(10, 360, 181, 50))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.kernel_size = QtWidgets.QLabel(self.layoutWidget2)
        self.kernel_size.setInputMethodHints(QtCore.Qt.ImhNone)
        self.kernel_size.setAlignment(QtCore.Qt.AlignCenter)
        self.kernel_size.setObjectName("kernel_size")
        self.verticalLayout_3.addWidget(self.kernel_size)
        self.slider = QtWidgets.QSlider(self.layoutWidget2)
        self.slider.setMinimum(3)
        self.slider.setSingleStep(2)
        self.slider.setTracking(True)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setObjectName("slider")
        self.verticalLayout_3.addWidget(self.slider)
        self.layoutWidget.raise_()
        self.layoutWidget.raise_()
        self.word2.raise_()
        self.word1.raise_()
        self.imageView.raise_()
        self.layoutWidget.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 986, 28))
        font = QtGui.QFont()
        font.setFamily("Kaiti SC")
        font.setPointSize(16)
        self.menubar.setFont(font)
        self.menubar.setTabletTracking(True)
        self.menubar.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.menubar.setAutoFillBackground(False)
        self.menubar.setStyleSheet("border-image: url(:/image/images/weather.jpg)")
        self.menubar.setDefaultUp(False)
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        self.geometry = QtWidgets.QMenu(self.menubar)
        self.geometry.setMouseTracking(True)
        self.geometry.setTabletTracking(True)
        self.geometry.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.geometry.setObjectName("geometry")
        self.histogram = QtWidgets.QMenu(self.menubar)
        self.histogram.setMouseTracking(True)
        self.histogram.setTabletTracking(True)
        self.histogram.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.histogram.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.histogram.setAcceptDrops(True)
        self.histogram.setTearOffEnabled(False)
        self.histogram.setSeparatorsCollapsible(False)
        self.histogram.setObjectName("histogram")
        self.smooth = QtWidgets.QMenu(self.menubar)
        self.smooth.setMouseTracking(True)
        self.smooth.setTabletTracking(True)
        self.smooth.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.smooth.setObjectName("smooth")
        self.sharpen = QtWidgets.QMenu(self.menubar)
        self.sharpen.setObjectName("sharpen")
        self.noise = QtWidgets.QMenu(self.menubar)
        self.noise.setObjectName("noise")
        self.frequency = QtWidgets.QMenu(self.menubar)
        self.frequency.setObjectName("frequency")
        self.file = QtWidgets.QMenu(self.menubar)
        self.file.setMouseTracking(True)
        self.file.setTabletTracking(True)
        self.file.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.file.setAutoFillBackground(False)
        self.file.setTearOffEnabled(False)
        self.file.setSeparatorsCollapsible(False)
        self.file.setToolTipsVisible(False)
        self.file.setObjectName("file")
        self.edgeDetection = QtWidgets.QMenu(self.menubar)
        self.edgeDetection.setObjectName("edgeDetection")
        self.gray = QtWidgets.QMenu(self.menubar)
        self.gray.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("SimSong")
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.gray.setFont(font)
        self.gray.setMouseTracking(True)
        self.gray.setTabletTracking(True)
        self.gray.setFocusPolicy(QtCore.Qt.NoFocus)
        self.gray.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.gray.setObjectName("gray")
        self.targetDetection = QtWidgets.QMenu(self.menubar)
        self.targetDetection.setObjectName("targetDetection")
        MainWindow.setMenuBar(self.menubar)
        self.open = QtWidgets.QAction(MainWindow)
        self.open.setObjectName("open")
        self.save = QtWidgets.QAction(MainWindow)
        self.save.setObjectName("save")
        self.close = QtWidgets.QAction(MainWindow)
        self.close.setObjectName("close")
        self.zoomIn = QtWidgets.QAction(MainWindow)
        self.zoomIn.setObjectName("zoomIn")
        self.zoomOut = QtWidgets.QAction(MainWindow)
        self.zoomOut.setObjectName("zoomOut")
        self.rotateL = QtWidgets.QAction(MainWindow)
        self.rotateL.setObjectName("rotateL")
        self.rotateR = QtWidgets.QAction(MainWindow)
        self.rotateR.setObjectName("rotateR")
        self.equalization = QtWidgets.QAction(MainWindow)
        self.equalization.setAutoRepeat(True)
        self.equalization.setObjectName("equalization")
        self.regulation = QtWidgets.QAction(MainWindow)
        self.regulation.setAutoRepeat(True)
        self.regulation.setObjectName("regulation")
        self.graying = QtWidgets.QAction(MainWindow)
        self.graying.setAutoRepeat(True)
        self.graying.setVisible(True)
        self.graying.setPriority(QtWidgets.QAction.LowPriority)
        self.graying.setObjectName("graying")
        self.reverse = QtWidgets.QAction(MainWindow)
        self.reverse.setAutoRepeat(True)
        self.reverse.setPriority(QtWidgets.QAction.LowPriority)
        self.reverse.setObjectName("reverse")
        self.binarization = QtWidgets.QAction(MainWindow)
        self.binarization.setAutoRepeat(True)
        self.binarization.setPriority(QtWidgets.QAction.NormalPriority)
        self.binarization.setObjectName("binarization")
        self.flipH = QtWidgets.QAction(MainWindow)
        self.flipH.setObjectName("flipH")
        self.flipR = QtWidgets.QAction(MainWindow)
        self.flipR.setObjectName("flipR")
        self.average = QtWidgets.QAction(MainWindow)
        self.average.setAutoRepeat(True)
        self.average.setObjectName("average")
        self.median = QtWidgets.QAction(MainWindow)
        self.median.setAutoRepeat(True)
        self.median.setObjectName("median")
        self.gaussian = QtWidgets.QAction(MainWindow)
        self.gaussian.setAutoRepeat(True)
        self.gaussian.setObjectName("gaussian")
        self.gaussianN = QtWidgets.QAction(MainWindow)
        self.gaussianN.setObjectName("gaussianN")
        self.impulseN = QtWidgets.QAction(MainWindow)
        self.impulseN.setObjectName("impulseN")
        self.randomN = QtWidgets.QAction(MainWindow)
        self.randomN.setObjectName("randomN")
        self.sobel = QtWidgets.QAction(MainWindow)
        self.sobel.setObjectName("sobel")
        self.robert = QtWidgets.QAction(MainWindow)
        self.robert.setObjectName("robert")
        self.prewitt = QtWidgets.QAction(MainWindow)
        self.prewitt.setObjectName("prewitt")
        self.laplacian = QtWidgets.QAction(MainWindow)
        self.laplacian.setObjectName("laplacian")
        self.bilateral = QtWidgets.QAction(MainWindow)
        self.bilateral.setObjectName("bilateral")
        self.highpass = QtWidgets.QAction(MainWindow)
        self.highpass.setObjectName("highpass")
        self.lowpass = QtWidgets.QAction(MainWindow)
        self.lowpass.setObjectName("lowpass")
        self.laplacianED = QtWidgets.QAction(MainWindow)
        self.laplacianED.setObjectName("laplacianED")
        self.sobelED = QtWidgets.QAction(MainWindow)
        self.sobelED.setObjectName("sobelED")
        self.cannyED = QtWidgets.QAction(MainWindow)
        self.cannyED.setObjectName("cannyED")
        self.yolov5 = QtWidgets.QAction(MainWindow)
        self.yolov5.setObjectName("yolov5")
        self.face = QtWidgets.QAction(MainWindow)
        self.face.setObjectName("face")
        self.segmentation = QtWidgets.QAction(MainWindow)
        self.segmentation.setObjectName("segmentation")
        self.hide = QtWidgets.QAction(MainWindow)
        self.hide.setObjectName("hide")
        self.color = QtWidgets.QAction(MainWindow)
        self.color.setObjectName("color")
        self.geometry.addAction(self.zoomIn)
        self.geometry.addAction(self.zoomOut)
        self.geometry.addAction(self.rotateL)
        self.geometry.addAction(self.rotateR)
        self.geometry.addAction(self.flipH)
        self.geometry.addAction(self.flipR)
        self.histogram.addAction(self.equalization)
        self.histogram.addAction(self.regulation)
        self.smooth.addAction(self.average)
        self.smooth.addAction(self.median)
        self.smooth.addAction(self.gaussian)
        self.smooth.addAction(self.bilateral)
        self.sharpen.addAction(self.sobel)
        self.sharpen.addAction(self.robert)
        self.sharpen.addAction(self.prewitt)
        self.sharpen.addAction(self.laplacian)
        self.sharpen.addAction(self.color)
        self.noise.addAction(self.gaussianN)
        self.noise.addAction(self.impulseN)
        self.noise.addAction(self.randomN)
        self.frequency.addAction(self.highpass)
        self.frequency.addAction(self.lowpass)
        self.file.addAction(self.open)
        self.file.addAction(self.save)
        self.file.addAction(self.close)
        self.file.addAction(self.hide)
        self.edgeDetection.addAction(self.laplacianED)
        self.edgeDetection.addAction(self.sobelED)
        self.edgeDetection.addAction(self.cannyED)
        self.gray.addAction(self.graying)
        self.gray.addAction(self.reverse)
        self.gray.addAction(self.binarization)
        self.targetDetection.addAction(self.yolov5)
        self.targetDetection.addAction(self.face)
        self.targetDetection.addAction(self.segmentation)
        self.menubar.addAction(self.file.menuAction())
        self.menubar.addAction(self.geometry.menuAction())
        self.menubar.addAction(self.gray.menuAction())
        self.menubar.addAction(self.histogram.menuAction())
        self.menubar.addAction(self.noise.menuAction())
        self.menubar.addAction(self.smooth.menuAction())
        self.menubar.addAction(self.sharpen.menuAction())
        self.menubar.addAction(self.frequency.menuAction())
        self.menubar.addAction(self.edgeDetection.menuAction())
        self.menubar.addAction(self.targetDetection.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Processing"))
        self.word1.setText(_translate("MainWindow", " Image Processing System"))
        self.word2.setText(_translate("MainWindow", "Show Image 👉"))
        self.viewOriginal.setText(_translate("MainWindow", "View Original"))
        self.drawHistogram.setText(_translate("MainWindow", "Draw Histogram"))
        self.brightness.setText(_translate("MainWindow", "Brightness Controller"))
        self.contrast.setText(_translate("MainWindow", "Contrast Controller"))
        self.kernel_size.setText(_translate("MainWindow", "Choose Your Kernel Size 👇"))
        self.geometry.setTitle(_translate("MainWindow", "几何变换"))
        self.histogram.setTitle(_translate("MainWindow", "直方图处理"))
        self.smooth.setTitle(_translate("MainWindow", "平滑处理"))
        self.sharpen.setTitle(_translate("MainWindow", "锐化处理"))
        self.noise.setTitle(_translate("MainWindow", "加性噪声"))
        self.frequency.setTitle(_translate("MainWindow", "频域滤波"))
        self.file.setTitle(_translate("MainWindow", "文件"))
        self.edgeDetection.setTitle(_translate("MainWindow", "边缘检测"))
        self.gray.setTitle(_translate("MainWindow", "灰度变换"))
        self.targetDetection.setTitle(_translate("MainWindow", "目标检测"))
        self.open.setText(_translate("MainWindow", "打开"))
        self.open.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.save.setText(_translate("MainWindow", "保存"))
        self.save.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.close.setText(_translate("MainWindow", "关闭"))
        self.close.setShortcut(_translate("MainWindow", "Ctrl+W"))
        self.zoomIn.setText(_translate("MainWindow", "放大"))
        self.zoomIn.setShortcut(_translate("MainWindow", "Meta+I"))
        self.zoomOut.setText(_translate("MainWindow", "缩小"))
        self.zoomOut.setShortcut(_translate("MainWindow", "Meta+O"))
        self.rotateL.setText(_translate("MainWindow", "向左旋转"))
        self.rotateL.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self.rotateR.setText(_translate("MainWindow", "向右旋转"))
        self.rotateR.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.equalization.setText(_translate("MainWindow", "直方图均衡化"))
        self.regulation.setText(_translate("MainWindow", "直方图规定化"))
        self.graying.setText(_translate("MainWindow", "图像灰度化"))
        self.reverse.setText(_translate("MainWindow", "图像反转"))
        self.binarization.setText(_translate("MainWindow", "图像二值化"))
        self.flipH.setText(_translate("MainWindow", "水平翻转"))
        self.flipR.setText(_translate("MainWindow", "垂直翻转"))
        self.average.setText(_translate("MainWindow", "均值滤波"))
        self.median.setText(_translate("MainWindow", "中值滤波"))
        self.gaussian.setText(_translate("MainWindow", "高斯滤波"))
        self.gaussianN.setText(_translate("MainWindow", "高斯噪声"))
        self.impulseN.setText(_translate("MainWindow", "椒盐噪声"))
        self.randomN.setText(_translate("MainWindow", "随机噪声"))
        self.sobel.setText(_translate("MainWindow", "sobel算子"))
        self.robert.setText(_translate("MainWindow", "robert算子"))
        self.prewitt.setText(_translate("MainWindow", "prewitt算子"))
        self.laplacian.setText(_translate("MainWindow", "laplacian算子"))
        self.bilateral.setText(_translate("MainWindow", "双边滤波"))
        self.highpass.setText(_translate("MainWindow", "高通滤波"))
        self.lowpass.setText(_translate("MainWindow", "低通滤波"))
        self.laplacianED.setText(_translate("MainWindow", "laplacian算子"))
        self.sobelED.setText(_translate("MainWindow", "sobel算子"))
        self.cannyED.setText(_translate("MainWindow", "canny算子"))
        self.yolov5.setText(_translate("MainWindow", "yolov5"))
        self.face.setText(_translate("MainWindow", "人脸检测"))
        self.segmentation.setText(_translate("MainWindow", "图像分割"))
        self.hide.setText(_translate("MainWindow", "隐藏"))
        self.hide.setShortcut(_translate("MainWindow", "Meta+H"))
        self.color.setText(_translate("MainWindow", "color"))
from imageview import ImageView
import res_rc

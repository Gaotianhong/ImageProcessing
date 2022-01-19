import sys
import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import exposure
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import mainwindow
from yolov5 import detect
from face_detection import detect_face_image, detect_face_video
from PIL import Image
from torchvision import transforms

matplotlib.use('MacOSX')
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()


def get_image_type(image):
    """获取图像类型(1)彩色图像 (2)灰度图像"""
    if len(image.shape) == 3:  # 三通道
        img_split = cv2.split(image)
        if (np.array(img_split[0]) == np.array(img_split[1])).all() and (
                np.array(img_split[1]) == np.array(img_split[2])).all():  # 三通道完全相同
            return 2  # 灰度图像
        else:
            return 1  # 彩色图像
    elif len(image.shape) == 2:  # 单通道
        return 2  # 灰度图像
    else:
        return -1


def show_image(image):
    """在画布上展示图像"""
    if get_image_type(image) == 1:  # 彩色图
        height, width, channels = image.shape
        ui_image = QImage(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width, height,
                          width * channels, QImage.Format_RGB888)
    else:  # 灰度图
        ui_image = QImage(image, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Indexed8)
    return ui_image


class ImageProcessing(mainwindow.Ui_MainWindow):
    def __init__(self, MainWindow):
        super().setupUi(MainWindow)

        self.file_name = None
        self.ap_file_name = None
        self.cv_srcImage = None
        self.cv_newImage = None

        self.imageScene = None
        self.pixmapItem = None

        self.filtering_items = [False] * 4  # 平滑处理

        self.init_ui()  # 响应

    def init_ui(self):
        self.showAll()  # 创建用于展示的场景
        self.hide_component()

        self.open.triggered.connect(self.open_file)  # 打开图片
        self.save.triggered.connect(self.save_file)  # 保存图片
        self.close.triggered.connect(self.close_file)  # 关闭图片
        self.hide.triggered.connect(self.hide_component)  # 关闭图片
        self.viewOriginal.clicked.connect(self.view_original)  # 显示原始图片

        self.zoomIn.triggered.connect(self.zoom_in_image)  # 放大图片
        self.zoomOut.triggered.connect(self.zoom_out_image)  # 缩小图片
        self.rotateL.triggered.connect(self.rotateL_image)  # 向左旋转图片
        self.rotateR.triggered.connect(self.rotateR_image)  # 向右旋转图片
        self.flipH.triggered.connect(self.flip_horizontal)  # 水平翻转图片
        self.flipR.triggered.connect(self.flip_vertical)  # 垂直翻转图片

        self.Ccontrast.valueChanged.connect(self.BC_controller)  # 图像对比度调节
        self.Cbrightness.valueChanged.connect(self.BC_controller)  # 图像亮度调节

        self.graying.triggered.connect(self.image_graying)  # 图像灰度化
        self.reverse.triggered.connect(self.image_reverse)  # 图像反转
        self.binarization.triggered.connect(self.image_binarization)  # 图像二值化

        self.equalization.triggered.connect(self.histogram_equalization)  # 直方图均衡化
        self.regulation.triggered.connect(self.histogram_specification)  # 直方图规定化
        self.drawHistogram.clicked.connect(self.draw_histogram)  # 绘制直方图

        self.gaussianN.triggered.connect(self.gaussian_noise)  # 高斯噪声
        self.impulseN.triggered.connect(self.impulse_noise)  # 椒盐噪声
        self.randomN.triggered.connect(self.random_noise)  # 随机噪声

        self.slider.valueChanged.connect(self.filtering)  # 卷积核大小
        self.average.triggered.connect(self.average_filtering)  # 均值滤波
        self.median.triggered.connect(self.median_filtering)  # 中值滤波
        self.gaussian.triggered.connect(self.gaussian_filter)  # 高斯滤波
        self.bilateral.triggered.connect(self.bilateral_filter)  # 双边滤波

        # 锐化处理
        self.sobel.triggered.connect(self.sobel_operator)  # sobel算子
        self.robert.triggered.connect(self.robert_operator)  # robert算子
        self.prewitt.triggered.connect(self.prewitt_operator)  # prewitt算子
        self.laplacian.triggered.connect(self.laplacian_operator)  # laplacian算子
        self.color.triggered.connect(self.color_format)  # color format

        self.highpass.triggered.connect(self.high_pass_filtering)  # 高通滤波
        self.lowpass.triggered.connect(self.low_pass_filtering)  # 低通滤波

        # 边缘检测
        self.laplacianED.triggered.connect(self.laplacian_edge_detection)  # laplacian算子
        self.sobelED.triggered.connect(self.sobel_edge_detection)  # sobel算子
        self.cannyED.triggered.connect(self.canny_edge_detection)  # canny算子

        self.yolov5.triggered.connect(self.yolov5_target_detection)  # yolov5目标检测
        self.face.triggered.connect(self.face_detection)  # 人脸检测
        self.segmentation.triggered.connect(self.image_segmentation)  # 图像分割

    def showAll(self):
        self.imageScene = QGraphicsScene()
        self.pixmapItem = QGraphicsPixmapItem()
        self.imageScene.addItem(self.pixmapItem)  # 添加画布
        self.imageView.setScene(self.imageScene)  # 设置场景

    def open_file(self):
        file_name, file_type = QFileDialog.getOpenFileName(QFileDialog(), '选择图片', 'test',
                                                           'Images (*.jpg *.png *.bmp *.jpeg)')
        self.ap_file_name = file_name
        self.file_name = file_name.split('/')[-1]
        if not file_name:
            return
        self.showAll()
        self.cv_srcImage = cv2.imread(file_name)
        cv2.imwrite('./yolov5/data/images/' + self.file_name, self.cv_srcImage)
        self.cv_newImage = self.cv_srcImage
        height, width, channels = self.cv_srcImage.shape
        ui_image = QImage(cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2RGB), width, height,
                          width * channels, QImage.Format_RGB888)
        self.imageView.resetTransform()  # 重置操作
        if ui_image.width() > self.imageView.width() or ui_image.height() > self.imageView.height():
            self.imageView.fitInView(QGraphicsPixmapItem(QPixmap(ui_image)))  # 适应场景大小
        self.pixmapItem.setPixmap(QPixmap(ui_image))

    def save_file(self):
        if self.pixmapItem.pixmap().isNull():
            return
        ui_image = self.pixmapItem.pixmap().toImage()
        file_name, file_type = QFileDialog.getSaveFileName(None, '保存图片', 'test/save_images/untitled', '*.jpg;;*.png')
        if not file_name:
            return
        ui_image.save(file_name)

    def close_file(self):
        self.cv_srcImage = None
        self.hide_component()
        self.pixmapItem.setPixmap(QPixmap())  # 设置空白画布

    def hide_component(self):
        self.kernel_size.hide()  # kernel_size隐藏
        self.slider.hide()  # 隐藏滤波滑动调节

    def view_original(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.hide_component()
        self.imageView.resetTransform()  # 重置操作
        self.cv_newImage = self.cv_srcImage
        ui_image = show_image(self.cv_srcImage)
        if ui_image.width() > self.imageView.width() or ui_image.height() > self.imageView.height():
            self.imageView.fitInView(QGraphicsPixmapItem(QPixmap(ui_image)))  # 适应场景大小
        self.pixmapItem.setPixmap(QPixmap(ui_image))

    def zoom_in_image(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.imageView.scale(1.2, 1.2)

    def zoom_out_image(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.imageView.scale(0.8, 0.8)

    def rotateL_image(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.imageView.rotate(-90.0)

    def rotateR_image(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.imageView.rotate(90.0)

    def flip_horizontal(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.imageView.scale(1, -1)
        self.imageView.rotate(-180.0)

    def flip_vertical(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.imageView.scale(-1, 1)
        self.imageView.rotate(-180.0)

    def BC_controller(self):
        if self.pixmapItem.pixmap().isNull():
            return
        img = self.cv_srcImage
        brightness = int(self.Cbrightness.value())
        contrast = int(self.Ccontrast.value())
        # print(brightness, contrast)
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                max = 255
            else:
                shadow = 0
                max = 255 + brightness
            al_pha = (max - shadow) / 255
            ga_mma = shadow
            # The function addWeighted calculates
            # the weighted sum of two arrays
            cal = cv2.addWeighted(img, al_pha,
                                  img, 0, ga_mma)
        else:
            cal = img
        if contrast != 0:
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)
            # The function addWeighted calculates
            # the weighted sum of two arrays
            cal = cv2.addWeighted(cal, Alpha,
                                  cal, 0, Gamma)
        self.cv_newImage = cal
        self.pixmapItem.setPixmap(QPixmap(show_image(cal)))

    def image_graying(self):
        if self.pixmapItem.pixmap().isNull():
            return
        if get_image_type(self.cv_srcImage) == 2:
            return
        self.cv_newImage = cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2GRAY)
        ui_image = QImage(self.cv_newImage, self.cv_newImage.shape[1], self.cv_newImage.shape[0],
                          self.cv_newImage.shape[1], QImage.Format_Grayscale8)  # 灰度化
        self.pixmapItem.setPixmap(QPixmap(ui_image))

    def image_reverse(self):
        if self.pixmapItem.pixmap().isNull():
            return
        ui_image = self.cv_srcImage
        if get_image_type(ui_image) == 1:  # 彩色图像
            height, width, channels = ui_image.shape
            dst = 255 - ui_image
            self.cv_newImage = dst
            dst = QImage(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), width, height,
                         width * channels, QImage.Format_RGB888)
            self.pixmapItem.setPixmap(QPixmap(dst))
        elif get_image_type(ui_image) == 2:  # 灰度图像或二值图像
            height = ui_image.shape[0]
            width = ui_image.shape[1]
            dst = 255 - ui_image
            self.cv_newImage = dst
            dst = QImage(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), width,
                         height, QImage.Format_Grayscale8)  # 灰度化
            self.pixmapItem.setPixmap(QPixmap(dst))
        else:
            return

    def image_binarization(self):
        if self.pixmapItem.pixmap().isNull():
            return
        gray = cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2GRAY)  # 灰度化
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化
        self.cv_newImage = binary
        ui_image = QImage(binary, binary.shape[1], binary.shape[0], binary.shape[1], QImage.Format_Indexed8)
        self.pixmapItem.setPixmap(QPixmap(ui_image))

    def histogram_equalization(self):
        if self.pixmapItem.pixmap().isNull():
            return
        if get_image_type(self.cv_srcImage) == 1:  # 彩色图
            (b, g, r) = cv2.split(self.cv_srcImage)
            bH = cv2.equalizeHist(b)
            gH = cv2.equalizeHist(g)
            rH = cv2.equalizeHist(r)
            dst = cv2.merge((bH, gH, rH))  # 合并每一个通道
            height, width, channels = dst.shape
            ui_image = QImage(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), width, height,
                              width * channels, QImage.Format_RGB888)
            self.pixmapItem.setPixmap(QPixmap(ui_image))
        else:  # 灰度图
            gray = cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2GRAY)
            dst = cv2.equalizeHist(gray)  # 直方图均衡化
            ui_image = QImage(dst, dst.shape[1], dst.shape[0], dst.shape[1], QImage.Format_Indexed8)
            self.pixmapItem.setPixmap(QPixmap(ui_image))
        self.cv_newImage = dst

    def histogram_specification(self):
        if self.pixmapItem.pixmap().isNull():
            return
        file_name, file_type = QFileDialog.getOpenFileName(QFileDialog(), '选择规定化的图片', 'test',
                                                           'Images (*.jpg *.png *.bmp *.jpeg)')
        if not file_name:
            return
        target = cv2.imread(file_name)
        multi = True if self.cv_srcImage.shape[-1] > 1 else False
        matched = exposure.match_histograms(self.cv_srcImage, target, multichannel=multi)
        self.cv_newImage = matched
        self.pixmapItem.setPixmap(QPixmap(show_image(matched)))

    def del_bug(self):
        self.file.destroy()
        self.geometry.destroy()
        self.gray.destroy()
        self.histogram.destroy()
        self.noise.destroy()
        self.smooth.destroy()
        self.sharpen.destroy()
        self.frequency.destroy()
        self.edgeDetection.destroy()
        self.targetDetection.destroy()

    def draw_histogram(self):
        if self.pixmapItem.pixmap().isNull():
            return
        plt.figure()
        self.del_bug()
        if get_image_type(self.cv_newImage) == 1:  # 彩色图
            color = ("b", "g", "r")
            for i, color in enumerate(color):
                hist = cv2.calcHist([self.cv_newImage], [i], None, [256], [0, 255])
                plt.plot(hist, color=color)
                plt.xlim([0, 256])
        else:  # 灰度图
            hist = cv2.calcHist([self.cv_newImage], [0], None, [256], [0, 255])
            plt.plot(hist)
        plt.show()

    def gaussian_noise(self):
        if self.pixmapItem.pixmap().isNull():
            return
        mean = 0  # 均值
        sigma = 0.1  # 方差
        img = self.cv_srcImage / 255  # 将图片灰度标准化
        noise = np.random.normal(mean, sigma, img.shape)  # 产生高斯 noise
        dst = img + noise  # 将噪声和图片叠加
        dst = np.clip(dst, 0, 1)  # 将超过 1 的置 1，低于 0 的置 0
        dst = np.uint8(dst * 255)  # 将图片灰度范围的恢复为 0-255
        self.cv_newImage = dst
        self.pixmapItem.setPixmap(QPixmap(show_image(dst)))

    def impulse_noise(self):
        if self.pixmapItem.pixmap().isNull():
            return
        img_noise = self.cv_srcImage.copy()
        height, width = img_noise.shape[0], img_noise.shape[1]  # 获取高度宽度像素值
        proportion = 0.1
        num = int(height * width * proportion)  # 噪声点数量
        for i in range(num):
            w = np.random.randint(0, width - 1)
            h = np.random.randint(0, height - 1)
            if np.random.randint(0, 1) == 0:
                img_noise[h, w] = 0
            else:
                img_noise[h, w] = 255
        self.cv_newImage = img_noise
        self.pixmapItem.setPixmap(QPixmap(show_image(img_noise)))

    def random_noise(self):
        if self.pixmapItem.pixmap().isNull():
            return
        img_noise = self.cv_srcImage.copy()
        height, width, channels = img_noise.shape
        noise_num = 1000  # 加噪声
        for i in range(noise_num):
            x = np.random.randint(0, height)  # 随机生成指定范围的整数
            y = np.random.randint(0, width)
            img_noise[x, y, :] = 0
        self.cv_newImage = img_noise
        self.pixmapItem.setPixmap(QPixmap(show_image(img_noise)))

    def filtering(self):
        if self.pixmapItem.pixmap().isNull():
            return
        kernel = int(self.slider.value())
        if self.filtering_items[0]:
            blur_img = cv2.blur(self.cv_srcImage, (kernel, kernel))  # 均值滤波
            self.cv_newImage = blur_img
            self.kernel_size.setText('均值滤波 Kernel Size:{}'.format(self.slider.value()))
            self.pixmapItem.setPixmap(QPixmap(show_image(blur_img)))
        elif self.filtering_items[1] and kernel % 2 != 0:
            blur_img = cv2.medianBlur(self.cv_srcImage, ksize=kernel)  # 中值滤波
            self.cv_newImage = blur_img
            self.kernel_size.setText('中值滤波 Kernel Size:{}'.format(self.slider.value()))
            self.pixmapItem.setPixmap(QPixmap(show_image(blur_img)))
        elif self.filtering_items[2] and kernel % 2 != 0:
            blur_img = cv2.GaussianBlur(self.cv_srcImage, (kernel, kernel), 1)  # 高斯滤波
            self.cv_newImage = blur_img
            self.kernel_size.setText('高斯滤波 Kernel Size:{}'.format(self.slider.value()))
            self.pixmapItem.setPixmap(QPixmap(show_image(blur_img)))

    def average_filtering(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.kernel_size.show()
        self.slider.show()
        self.kernel_size.setText('均值滤波 Kernel Size:👇')
        self.filtering_items = [True if i == 0 else False for i in range(len(self.filtering_items))]

    def median_filtering(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.kernel_size.show()
        self.slider.show()
        self.kernel_size.setText('中值滤波 Kernel Size:👇')
        self.filtering_items = [True if i == 1 else False for i in range(len(self.filtering_items))]

    def gaussian_filter(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.kernel_size.show()
        self.slider.show()
        self.kernel_size.setText('高斯滤波 Kernel Size:👇')
        self.filtering_items = [True if i == 2 else False for i in range(len(self.filtering_items))]

    def bilateral_filter(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.kernel_size.show()
        self.slider.hide()
        self.kernel_size.setText('双边滤波')
        blur_img = cv2.bilateralFilter(self.cv_srcImage, 30, sigmaColor=200, sigmaSpace=10)
        self.cv_newImage = blur_img
        self.pixmapItem.setPixmap(QPixmap(show_image(blur_img)))

    def sobel_operator(self):
        if self.pixmapItem.pixmap().isNull():
            return
        # sobel算子
        x = cv2.Sobel(self.cv_srcImage, -1, 1, 0)
        y = cv2.Sobel(self.cv_srcImage, -1, 0, 1)
        absX = cv2.convertScaleAbs(x)  # 转回unit8
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        self.cv_newImage = dst
        self.pixmapItem.setPixmap(QPixmap(show_image(dst)))

    def robert_operator(self):
        if self.pixmapItem.pixmap().isNull():
            return
        # Roberts算子
        kernelX = np.array([[-1, 0], [0, 1]], dtype=int)
        kernelY = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv2.filter2D(self.cv_srcImage, -1, kernelX)
        y = cv2.filter2D(self.cv_srcImage, -1, kernelY)
        absX = cv2.convertScaleAbs(x)  # 转uint8
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        self.cv_newImage = dst
        self.pixmapItem.setPixmap(QPixmap(show_image(dst)))

    def prewitt_operator(self):
        if self.pixmapItem.pixmap().isNull():
            return
        # prewitt算子
        kernelX = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernelY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv2.filter2D(self.cv_srcImage, -1, kernelX)
        y = cv2.filter2D(self.cv_srcImage, -1, kernelY)
        absX = cv2.convertScaleAbs(x)  # 转uint8
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        self.cv_newImage = dst
        self.pixmapItem.setPixmap(QPixmap(show_image(dst)))

    def laplacian_operator(self):
        if self.pixmapItem.pixmap().isNull():
            return
        dst = cv2.Laplacian(self.cv_srcImage, -1, ksize=3)
        Laplacian = cv2.convertScaleAbs(dst)
        self.cv_newImage = Laplacian
        self.pixmapItem.setPixmap(QPixmap(show_image(Laplacian)))

    def color_format(self):
        if self.pixmapItem.pixmap().isNull():
            return
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        image_sharp = cv2.filter2D(src=self.cv_srcImage, ddepth=-1, kernel=kernel)
        self.cv_newImage = image_sharp
        self.pixmapItem.setPixmap(QPixmap(show_image(image_sharp)))

    def high_pass_filtering(self):
        if self.pixmapItem.pixmap().isNull():
            return
        gaussBlur = cv2.GaussianBlur(self.cv_srcImage, (5, 5), cv2.BORDER_DEFAULT)
        highPass = self.cv_srcImage - gaussBlur
        # highPass = highPass + 127 * np.ones(self.cv_srcImage.shape, np.uint8)
        self.cv_newImage = highPass
        self.pixmapItem.setPixmap(QPixmap(show_image(highPass)))

    def low_pass_filtering(self):
        if self.pixmapItem.pixmap().isNull():
            return
        kernel = np.ones((10, 10), np.float32) / 25
        lowPass = cv2.filter2D(self.cv_srcImage, -1, kernel)
        lowPass = self.cv_srcImage - lowPass
        self.cv_newImage = lowPass
        self.pixmapItem.setPixmap(QPixmap(show_image(lowPass)))

    def laplacian_edge_detection(self):
        if self.pixmapItem.pixmap().isNull():
            return
        img = cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2GRAY)
        gaussBlur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
        gray_lap = cv2.Laplacian(gaussBlur, cv2.CV_16S, ksize=3)
        dst = cv2.convertScaleAbs(gray_lap)
        self.pixmapItem.setPixmap(QPixmap(show_image(dst)))

    def sobel_edge_detection(self):
        if self.pixmapItem.pixmap().isNull():
            return
        img = cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2GRAY)
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        self.pixmapItem.setPixmap(QPixmap(show_image(dst)))

    def canny_edge_detection(self):
        if self.pixmapItem.pixmap().isNull():
            return
        img = cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2GRAY)
        gaussBlur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
        gray = cv2.Canny(gaussBlur, 30, 80)
        dst = cv2.convertScaleAbs(gray)
        self.pixmapItem.setPixmap(QPixmap(show_image(dst)))

    def yolov5_target_detection(self):
        if self.pixmapItem.pixmap().isNull():
            return
        opt = detect.parse_opt()
        detect.main(opt)
        self.cv_srcImage = cv2.imread('./yolov5/runs/' + self.file_name)
        height, width, channels = self.cv_srcImage.shape
        ui_image = QImage(cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2RGB), width, height,
                          width * channels, QImage.Format_RGB888)
        if ui_image.width() > self.imageView.width() or ui_image.height() > self.imageView.height():
            self.imageView.fitInView(QGraphicsPixmapItem(QPixmap(ui_image)))  # 适应场景大小
        self.pixmapItem.setPixmap(QPixmap(ui_image))

    def face_detection(self):
        if self.pixmapItem.pixmap().isNull():
            detect_face_video.run()
        else:
            img = detect_face_image.run(self.cv_srcImage)
            self.pixmapItem.setPixmap(QPixmap(show_image(img)))

    def image_segmentation(self):
        if self.pixmapItem.pixmap().isNull():
            return
        self.del_bug()
        filename = "deeplab1.png"
        input_image = Image.open(filename)
        input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)
        plt.imshow(r)
        plt.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = ImageProcessing(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

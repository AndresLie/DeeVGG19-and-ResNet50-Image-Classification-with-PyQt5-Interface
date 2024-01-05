
from PyQt5.QtWidgets import QDialog,QSizePolicy,QVBoxLayout
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

class drw_contour(QDialog):
    def __init__(self,image_path):
        super().__init__()
        self.image_path=image_path
        image=cv2.imread(self.image_path)
        image_RGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_gray=cv2.cvtColor(image_RGB,cv2.COLOR_RGB2GRAY)
        image_gray=cv2.GaussianBlur(image_gray,(5,5),0)
        circles = cv2.HoughCircles(
            image_gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=50
        )
        image_rgb_circle=image_RGB.copy()
        image3=image_gray.copy()
        image3[:,:]=0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(image_rgb_circle, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(image3, (i[0], i[1]), 2, 255, 3)
        if circles is not None:
            self.total_coins = len(circles[0])
        fig = plt.figure(figsize=(7, 5))
        plt.subplot(131)
        plt.imshow(image_RGB)
        plt.subplot(132)
        plt.imshow(image_rgb_circle)
        plt.subplot(133)
        plt.imshow(image3)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        sub_dialog = QDialog(self)
        sub_dialog.setLayout(layout)
        sub_dialog.setWindowTitle("1.1")
        sub_dialog.exec_() 
    def get_coin(self):
        return self.total_coins
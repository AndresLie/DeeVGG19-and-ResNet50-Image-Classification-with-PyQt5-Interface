from PyQt5.QtWidgets import  QDialog, QVBoxLayout,QSizePolicy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class hist_equ(QDialog):
    def __init__(self,image_path):
        super().__init__()
        self.image_path=image_path
        image=cv2.imread(self.image_path)
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist_gray=np.array(image_gray)
        hist_gray=hist_gray.flatten()
        cv2_equalized_image = cv2.equalizeHist(image_gray)
        hist_cv2=np.array(cv2_equalized_image)
        hist_cv2=hist_cv2.flatten()

        hist, bins = np.histogram(image_gray.flatten(), bins=256, range=(0, 256), density=True)

       
        cdf = hist.cumsum()


        manual_equalized_image= np.interp(image_gray.flatten(), bins[:-1], cdf * 255).reshape(image_gray.shape).astype(np.uint8)
        hist_manual=np.array(manual_equalized_image)
        hist_manual=hist_manual.flatten()
     

        fig=plt.figure(figsize=(9,7))
        plt.subplot(231)
        plt.imshow(image_gray,cmap='gray')
        plt.subplot(232)
        plt.imshow(cv2_equalized_image,cmap='gray')
        plt.subplot(233)
        plt.imshow(manual_equalized_image,cmap='gray')
        plt.subplot(235)
        plt.hist(hist_cv2,bins=256, range=(0, 256),)
        plt.subplot(234)
        plt.hist(hist_gray,bins=256, range=(0, 256),)
        plt.subplot(236)
        plt.hist(hist_manual,bins=256, range=(0, 256),)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        sub_dialog = QDialog(self)
        sub_dialog.setLayout(layout)
        sub_dialog.setWindowTitle("1.1")
        sub_dialog.exec_() 
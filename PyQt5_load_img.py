from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap   
import cv2
from PIL import Image

def Qt_load_img(label=None,return_path=False):
    
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_dialog = QFileDialog(options=options)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec_():
            image_path = file_dialog.selectedFiles()[0]
            img=cv2.imread(image_path)
            pixmap = QPixmap(image_path)
            if label:
                label.setPixmap(pixmap)
            if return_path:
                 return image_path
            return Image.open(image_path)

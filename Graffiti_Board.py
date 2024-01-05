from PyQt5.QtGui import QPixmap, QImage,QPainter,QPen,QPalette
from PyQt5.QtWidgets import QVBoxLayout, QWidget,QLabel
from PyQt5.QtCore import Qt
import numpy as np

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.resize(500, 500)
        self.setWindowTitle('Painter Board')
        self.tracing_xy = []
        self.lineHistory = []
        self.pen = QPen(Qt.white, 10, Qt.SolidLine)  # Set pen color to white

        # Set the background color of the central widget to black
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.black)
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        # Create a QLabel to display the image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Create a layout to add the image label
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)

    def get_image(self, grayscale=False):
        # Create a QPixmap to store the contents of the widget
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.black)  # Fill the pixmap with a black background

        # Create a QPainter to paint on the pixmap
        painter = QPainter(pixmap)
        painter.setPen(self.pen)

        # Draw the lines stored in lineHistory on the pixmap
        for line in self.lineHistory:
            for point_n in range(1, len(line)):
                start_x, start_y = line[point_n - 1][0], line[point_n - 1][1]
                end_x, end_y = line[point_n][0], line[point_n][1]
                painter.drawLine(start_x, start_y, end_x, end_y)

        # Draw the currently traced lines on the pixmap
        start_x_temp, start_y_temp = 0, 0
        for x, y in self.tracing_xy:
            if start_x_temp == 0 and start_y_temp == 0:
                painter.drawLine(self.start_xy[0][0], self.start_xy[0][1], x, y)
            else:
                painter.drawLine(start_x_temp, start_y_temp, x, y)

            start_x_temp = x
            start_y_temp = y

        painter.end()  # End painting on the pixmap

        # Convert the pixmap to a QImage
        image = pixmap.toImage()

        # Convert the QImage to a NumPy array
        if grayscale:
            image = image.convertToFormat(QImage.Format_Grayscale8)
            width, height = image.width(), image.height()
            ptr = image.bits()
            ptr.setsize(width * height)
            array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width))
        else:
            buffer = image.bits()
            buffer.setsize(image.byteCount())
            array = np.frombuffer(buffer, dtype=np.uint8).reshape((image.height(), image.width(), 4))

        return array
    
    def paintEvent(self, QPaintEvent):
         self.painter = QPainter()
         self.painter.begin(self)
         self.painter.setPen(self.pen)

         start_x_temp = 0
         start_y_temp = 0

         if self.lineHistory:
             for line_n in range(len(self.lineHistory)):
                 for point_n in range(1, len(self.lineHistory[line_n])):
                     start_x, start_y = self.lineHistory[line_n][point_n-1][0], self.lineHistory[line_n][point_n-1][1]
                     end_x, end_y = self.lineHistory[line_n][point_n][0], self.lineHistory[line_n][point_n][1]
                     self.painter.drawLine(start_x, start_y, end_x, end_y)

         for x, y in self.tracing_xy:
             if start_x_temp == 0 and start_y_temp == 0:
                 self.painter.drawLine(self.start_xy[0][0], self.start_xy[0][1], x, y)
             else:
                 self.painter.drawLine(start_x_temp, start_y_temp, x, y)

             start_x_temp = x
             start_y_temp = y

         self.painter.end()

    def mousePressEvent(self, QMouseEvent):
         self.start_xy = [(QMouseEvent.pos().x(), QMouseEvent.pos().y())]

    def mouseMoveEvent(self, QMouseEvent):
         self.tracing_xy.append((QMouseEvent.pos().x(), QMouseEvent.pos().y()))
         self.update()

    def mouseReleaseEvent(self, QMouseEvent):
         self.lineHistory.append(self.start_xy+self.tracing_xy)
         self.tracing_xy = []
    def reset(self):
        # Clear the line history
        self.image_label.clear()
        self.lineHistory = []
        self.update()

    def set_image(self, img_path):
        # Load the image from file
        pixmap = QPixmap(img_path)

        # Set the pixmap to the image label
        self.image_label.setPixmap(pixmap)
import sys, torch,random,cv2,torchvision.transforms as transforms,torch.nn as nn,os,matplotlib.pyplot as plt,torchvision
from PyQt5.QtWidgets import QApplication, QDialog, QGridLayout, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use("Qt5Agg") 
from PIL import Image
from torchsummary import summary
from torchvision.models import vgg19_bn
from PyQt5_load_img import Qt_load_img
from Graffiti_Board import MainWindow
from draw_contour import drw_contour
from hist_equ import hist_equ
from Morphology_Operation import Opening,Closing
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class CombinedDialog(QDialog):
    def __init__(self):
        
        super().__init__()
        self.image_path=None
        
        self.setWindowTitle("Combined Image Processing")
        self.setGeometry(130, 130, 1800, 600)

        layout = QGridLayout()

        # Left Part
        left_layout = QVBoxLayout()
        self.image_path_label = QLabel("Image 1 Path:")
        left_layout.addWidget(self.image_path_label)

       

        image1_button = QPushButton("Load Image")
        image1_button.clicked.connect(self.upload_image)
        left_layout.addWidget(image1_button,alignment=Qt.AlignCenter)

 

        layout.addLayout(left_layout, 0, 0)

        # Middle Part
        middle_layout = QVBoxLayout()

        # Sub-Part 1
        part1_label = QLabel("Hough Image Transform")
        middle_layout.addWidget(part1_label)

        self.button1 = QPushButton("1.1 Draw Contour")
        self.button1.clicked.connect(self.draw_contour)
        middle_layout.addWidget(self.button1)

        self.button2 = QPushButton("1.2 Count Coin")
        self.button2.clicked.connect(self.count_coin)
        middle_layout.addWidget(self.button2)

        self.label1=QLabel("There are _ coins in the image")
        middle_layout.addWidget(self.label1)

        # # Sub-Part 2
        part2_label = QLabel("2. Histogram Equalization")
        middle_layout.addWidget(part2_label)

        self.button4 = QPushButton("2. Histogram Equalization")
        self.button4.clicked.connect(self.hist_equ)
        middle_layout.addWidget(self.button4)
        
        # # Sub-Part 3
        part3_label = QLabel("3. Morphology Operation")
        middle_layout.addWidget(part3_label)

        self.button7 = QPushButton("3.1 Closing")
        self.button7.clicked.connect(self.open_closing)
        middle_layout.addWidget(self.button7)

        self.button8 = QPushButton("3.2 Opening")
        self.button8.clicked.connect(self.open_opening)
        middle_layout.addWidget(self.button8)
        layout.addLayout(middle_layout, 0, 1)

        # # Right Part
        right_layout = QVBoxLayout()
        container = QWidget()
        container.setStyleSheet("background-color: black;")
        # Sub-Part 4
        part4_label = QLabel("MNIST Classfier Using VGG-19")
        right_layout.addWidget(part4_label)
        part4=QHBoxLayout()
        part4_button=QVBoxLayout()
        button_vgg_struct_1=QPushButton("1. Show Model Structure")
        button_vgg_struct_1.clicked.connect(self.vgg_structure)
        part4_button.addWidget(button_vgg_struct_1)
        button_vgg_acc=QPushButton("2. Show Accuracy and Loss")
        button_vgg_acc.clicked.connect(self.vgg_acc_loss)
        part4_button.addWidget(button_vgg_acc)
        button_vgg_predict=QPushButton("3. Predict")
        button_vgg_predict.clicked.connect(self.vgg_predict)
        part4_button.addWidget(button_vgg_predict)
        button_vgg_reset=QPushButton("4. Reset")
        button_vgg_reset.clicked.connect(self.vgg_reset)
        part4_button.addWidget(button_vgg_reset)
        self.vgg_predict=QLabel("")
        self.vgg_predict.setFixedSize(140,40)
        self.vgg_predict.setAlignment(Qt.AlignCenter)
        part4_button.addWidget(self.vgg_predict)
        self.board = MainWindow()
        part4.addLayout(part4_button)
        part4.addWidget(self.board)
        right_layout.addLayout(part4)    

        part5=QHBoxLayout()
        part5_button=QVBoxLayout()
        part5_label=QLabel("ResNet50")
        part5_button.addWidget(part5_label)
        button_resnet_load_img=QPushButton("Load Image")
        button_resnet_load_img.clicked.connect(self.resnet_load_img)
        part5_button.addWidget(button_resnet_load_img)
        button_resnet_img__sample=QPushButton("5.1 Show Image")
        button_resnet_img__sample.clicked.connect(self.resnet_show_img)
        part5_button.addWidget(button_resnet_img__sample)
        button_resnet_mdl_struct=QPushButton("5.2 Show Model Structure")
        button_resnet_mdl_struct.clicked.connect(self.resnet_struct)
        part5_button.addWidget(button_resnet_mdl_struct)
        button_resnet_show_compare=QPushButton("5.3 Show Comparison")
        button_resnet_show_compare.clicked.connect(self.resnet_compare)
        part5_button.addWidget(button_resnet_show_compare)
        button_resnet_inference=QPushButton("5.4 Inference")
        button_resnet_inference.clicked.connect(self.resnet_inference)
        part5_button.addWidget(button_resnet_inference)
        self.resnet_result_lbl=QLabel("Prediction : ")
        self.resnet_result_lbl.setFixedSize(140,40)
        self.resnet_result_lbl.setAlignment(Qt.AlignCenter)
        part5_button.addWidget(self.resnet_result_lbl)
        self.resnet_img_lbl=QLabel("")
        label_width = 400  # Change to your desired width
        label_height = 300  # Change to your desired height
        self.resnet_img_lbl.setFixedSize(label_width, label_height)
        self.resnet_img_lbl.setScaledContents(True)
        self.resnet_img_lbl.setAlignment(Qt.AlignCenter)  # Center-align the text
        
        
        # Add a border using style sheets
        self.resnet_img_lbl.setStyleSheet("QLabel { border: 2px solid black; }")
        part5.addLayout(part5_button)
        part5.addWidget(self.resnet_img_lbl)
        right_layout.addLayout(part5)

        layout.addLayout(right_layout, 0, 2)

        self.setLayout(layout)
        self.model_vgg = vgg19_bn(num_classes=10)

        # Load the pre-trained weights
        state_dict = torch.load('vgg19_bn_mnist.pth')
        
        # Load the state dictionary into the model
        self.model_vgg.load_state_dict(state_dict)

        # Set the model to evaluation mode
        self.model_vgg.eval()

        self.model_resnet=torchvision.models.resnet50()
        self.model_resnet.fc = nn.Linear(2048, 1)
        state_dict=torch.load('resnet50_model.pth')
        adapted_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("fc.0.", "fc.")
            adapted_state_dict[new_key] = value

        self.model_resnet.load_state_dict(adapted_state_dict)
        self.model_resnet.eval()

    def upload_image(self):
        self.image_path=Qt_load_img(return_path=True)
        self.image_path_label.setText(f"\nImage  Path: {self.image_path}")

    def draw_contour(self):
        if self.image_path:
            a=drw_contour(self.image_path)
            self.coin=a.get_coin()
    
    def count_coin(self):
        self.label1.setText("There are {} coins in the image".format(self.coin))

    def hist_equ(self):
        if self.image_path:
            hist_equ(self.image_path)
    
    def open_closing(self):
        if self.image_path:
            Closing(self.image_path)

    def open_opening(self):
        if self.image_path:
            Opening(self.image_path)
     
    def vgg_structure(self):
        summary(self.model_vgg, input_size=(3,32,32))

    def vgg_acc_loss(self):
 
        self.board.set_image("vgg19_bn_mnist_results.png")
        self.board.show()

    def vgg_predict(self):
        class_labels=[i for i in range(0,10)]
        img=self.board.get_image()
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img=Image.fromarray(img)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB (3 channels)
            transforms.ToTensor(),
        ])
        input_tensor = transform(img)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.model_vgg(input_batch)

        # The output is a tensor with raw scores for each class; use a softmax function to get probabilities
        import torch.nn.functional as F
        probabilities = F.softmax(output, dim=1)
        fig=plt.figure()
        plt.bar(range(len(class_labels)), probabilities[0].numpy()[:len(class_labels)])
        plt.xticks(class_labels)
        plt.xlabel('Class')
        plt.ylabel('Probability')
        canvas=FigureCanvas(fig)
        layout=QVBoxLayout()
        layout.addWidget(canvas)
        sub_dialog = QDialog(self)
        sub_dialog.setLayout(layout)
        sub_dialog.exec_() 
        predicted_class = probabilities.argmax().item()
        self.vgg_predict.setText(str(predicted_class))

    def vgg_reset(self):
        self.board.reset()

    def resnet_load_img(self):
        self.resnet_img=None
        self.resnet_img=Qt_load_img(self.resnet_img_lbl)
       
    def resnet_show_img(self):

        path1='inference_dataset/Cat'
        path2='inference_dataset/Dog'

        dog_photo_name=[]
        cat_photo_name=[]
        dog_photo_name=os.listdir(path2)
        cat_photo_name=os.listdir(path1)
        img1=os.path.join(path2,dog_photo_name[random.randint(0,4)])
        img2=os.path.join(path1,cat_photo_name[random.randint(0,4)])
        cat_img=Image.open(img2)
        dog_img=Image.open(img1)
        fig=plt.figure()
        plt.subplot(121)
        plt.imshow(cat_img)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(122)
        plt.imshow(dog_img)
        plt.xticks([])
        plt.yticks([])
        canvas=FigureCanvas(fig)
        layout=QVBoxLayout()
        layout.addWidget(canvas)
        sub_dialog = QDialog(self)
        sub_dialog.setLayout(layout)
        sub_dialog.exec_() 

    def resnet_struct(self):
        summary(self.model_resnet, input_size=(3,224,224))

    def resnet_compare(self):
        import pandas as pd 
        resnet_ori=pd.read_csv('res50_val_acc.csv')
        resnet_RandomErasing=pd.read_csv("res50__rndmerasing_val_acc.csv")
        resnet_ori=resnet_ori.iloc[-1,1]*100
        resnet_RandomErasing=resnet_RandomErasing.iloc[-1,1]*100
        y_axis=[resnet_ori,resnet_RandomErasing]
        # y_axis=y_axis*100
        x_axis=["Without Random Erasing","With Random Erasing"]
        fig=plt.figure()
        plt.bar(x_axis,y_axis)
        for i, value in enumerate(y_axis):
            plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')
        canvas=FigureCanvas(fig)
        layout=QVBoxLayout()
        layout.addWidget(canvas)
        sub_dialog = QDialog(self)
        sub_dialog.setLayout(layout)
        sub_dialog.exec_() 

    def resnet_inference(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        input_tensor=transform(self.resnet_img)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.model_resnet(input_batch)
        # The output is a tensor with raw scores for each class; use a softmax function to get probabilities
        import torch.nn.functional as F
        probabilities = F.sigmoid(output)
        threshold=0.5
        if probabilities >threshold:
            predicted_class='Dog'
        else:
            predicted_class='Cat'
        self.resnet_result_lbl.setText("Predictions : {}".format(predicted_class))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = CombinedDialog()
    dialog.exec_()
    sys.exit(app.exec_())

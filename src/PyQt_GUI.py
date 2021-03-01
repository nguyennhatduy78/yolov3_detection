from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import cv2
import utils as utils
from PyQt5.QtCore import *
import numpy as np


# GUI initial
class VideoThread(QThread):
    signal_out = pyqtSignal(QPixmap)
    info_out = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.path = ''
        self._run_flag = True

    def set_path(self, path_):
        self.path = path_

    def run(self):
        if self.path == '':
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            while self._run_flag:
                ret, frame = cap.read()
                w = int(frame.shape[0])
                h = int(frame.shape[1])
                if w > 1024 or w > 768:
                    w = int(w*0.5)
                    h = int(h*0.5)
                    frame = cv2.resize(frame,(h,w))
                if ret:
                    frame, info = Run(frame)
                    self.signal_out.emit(frame)
                    self.info_out.emit(info)
        if self.path != '':
            cap = cv2.VideoCapture(self.path)
            while self._run_flag:
                ret, frame = cap.read()
                w = int(frame.shape[0])
                h = int(frame.shape[1])
                if w > 1024 or w > 768:
                    w = int(w*0.5)
                    h = int(h*0.5)
                    frame = cv2.resize(frame,(h,w))
                if ret:
                    frame, info = Run(frame)
                    self.signal_out.emit(frame)
                    self.info_out.emit(info)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.width = 1280
        self.height = 720
        self.webcam = QLabel(self)
        self.video = QLabel(self)
        self.image = QLabel(self)
        self.webcam.resize(self.width, self.height)
        self.exit_btn = QPushButton('Exit')
        self.error_message = QMessageBox()

        self.type_media = QComboBox()
        self.type_media.setFixedSize(100, 30)
        self.type_media.addItem('Webcam')
        self.type_media.addItem('Image')
        self.type_media.addItem('Video')

        self.stack1 = QWidget()
        self.stack2 = QWidget()
        self.stack3 = QWidget()

        self.stack1UI()
        self.stack2UI()
        self.stack3UI()

        self.Stack = QStackedWidget(self)
        self.Stack.addWidget(self.stack1)
        self.Stack.addWidget(self.stack2)
        self.Stack.addWidget(self.stack3)

        layout = QGridLayout(self)
        label_ = QLabel('YOLO V3 DEMO')
        sub_label = QLabel('INFORMATION')

        self.error_message.setWindowTitle('ERROR')
        self.error_message.setText('Please choose a file, or switch to Webcam to use Webcam')
        self.error_message.setIcon(QMessageBox.Critical)
        self.info_pane = QLabel()
        self.info_pane.setWordWrap(True)
        self.info_pane.setFixedSize(800, 600)
        self.info_pane.setAlignment(Qt.AlignTop)
        label_.setFont(QFont('Times', 35))
        sub_label.setFont(QFont('Times', 24))
        self.info_pane.setFont(QFont('Times', 18))
        layout.addWidget(label_, 0, 0)
        layout.addWidget(self.type_media, 1, 0)
        layout.addWidget(self.Stack, 2, 0, 10, 10)
        layout.addWidget(sub_label, 2, 14)
        layout.addWidget(self.info_pane, 1, 10, 10, 10)
        layout.addWidget(self.exit_btn, 12, 0)
        self.setLayout(layout)
        self.type_media.activated.connect(self.display)
        self.exit_btn.clicked.connect(lambda: self.close())
        self.media_thread = self.thread()
        self.setGeometry(300, 50, 10, 10)
        self.setWindowTitle('Project III - 20201')
        self.show()

    def stack1UI(self):
        layout = QGridLayout()
        start_ = QPushButton('Start')
        stop_ = QPushButton('Stop')
        layout.addWidget(start_, 0, 0)
        layout.addWidget(stop_, 0, 1)
        layout.addWidget(self.webcam, 1, 0, 8, 8)
        start_.clicked.connect(self.webcam_on)
        stop_.clicked.connect(self.media_thread_off)
        self.stack1.setLayout(layout)

    def stack2UI(self):
        layout = QGridLayout()
        open_btn = QPushButton('Choose image file')
        open_btn.clicked.connect(self.get_image)
        layout.addWidget(open_btn, 0, 0, 1, 4)
        layout.addWidget(self.image, 1, 0, 8, 8)
        self.stack2.setLayout(layout)

    def stack3UI(self):
        layout = QGridLayout()
        open_btn = QPushButton('Choose video file')
        stop_btn = QPushButton('Stop')
        open_btn.clicked.connect(self.get_video)
        stop_btn.clicked.connect(self.media_thread_off)
        layout.addWidget(open_btn, 0, 0, 1, 4)
        layout.addWidget(stop_btn, 0, 4, 1, 2)
        layout.addWidget(self.video, 1, 0, 8, 8)
        self.stack3.setLayout(layout)

    def display(self, i):
        self.Stack.setCurrentIndex(i)

    def set_info(self, info_list):
        tmp = ""
        for x in info_list:
            if not tmp.__contains__(x) or x == '---':
                tmp += '\n' + x
        self.info_pane.setText(tmp)

    def get_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Choose image file', '../data', "Image files (*.jpg *png)")
        print(str(fname))
        if fname[0] == '':
            self.error_message.show()
        if fname[0] != '':
            img = cv2.imread(fname[0])
            w = int(img.shape[0])
            h = int(img.shape[1])
            if w > 1024 or w > 768:
                w = int(w*0.5)
                h = int(h*0.5)
                img = cv2.resize(img,(h,w))
            result_img, info = Run(img)
            self.image.setPixmap(result_img)
            self.set_info(info)

    def get_video(self):
        fname = QFileDialog.getOpenFileName(self, 'Choose video file', '../data', "Video files (*.mp4 *avi)")
        print(fname[0])
        if fname[0] == '':
            self.error_message.show()
        if fname[0] != '':
            self.media_thread = VideoThread()
            self.media_thread.set_path(fname[0])
            self.media_thread.signal_out.connect(self.video_render)
            self.media_thread.info_out.connect(self.set_info)
            self.media_thread.start()

    def video_render(self, cv_img):
        self.video.setPixmap(cv_img)

    def webcam_on(self):
        self.media_thread = VideoThread()
        self.media_thread.signal_out.connect(self.webcam_render)
        self.media_thread.info_out.connect(self.set_info)
        self.media_thread.start()

    def media_thread_off(self):
        self.media_thread.stop()

    def webcam_render(self, cv_img):
        self.webcam.setPixmap(cv_img)


# Yolo v3 initial

# Define dir
class_dir = '../models/HungDuy.txt'
weights_dir = '../models/HungDuy.weights'

# Define param for yoloV3
class_names = [c.strip() for c in open(class_dir).readlines()]

channels = 3

class_num = len(class_names)

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

# Initiate yolo function
yolo = utils.YoloV3(None, channels, yolo_anchors, yolo_anchor_masks, class_num)
utils.load_darknet_weights(yolo, weights_dir)

# Ultilities

MySQL_open = utils.MySQL('localhost', 'nguyenduy2911', 'Nh@tduy1998', 'project3')
student_list_ = MySQL_open.get_all()


def Run(image__):
    img = utils.img_process(image__)
    boxes, scores, classes, nums = yolo(img)
    image__ = utils.draw_outputs(image__, (boxes, scores, classes, nums), class_names)
    count = 0
    obj_ = []
    values_ = []
    for i in range(class_num):
        obj_.append(0)
    for i in range(nums[0]):
        idx = int(classes[0][i])
        obj_[idx] += 1
    for x in range(len(obj_)):
        if obj_[x] != 0:
            count += 1
            print(class_names[x])
            for y in student_list_:
                if y[1] == class_names[x]:
                    print(class_names[x])
                    values_ += [
                        'Object : ' + y[1],
                        'Student : ' + y[2],
                        'Age : ' + str(y[3]),
                        'Gmail: ' + y[4],
                        'Description : ' + y[5],
                        '---------------'
                    ]
    if count == 0:
        values_ = ['There is no object detected !!!']
    if count != 0:
        values_ += [
            'TOTAL STUDENTS : ' + str(count)
        ]
    # print(sys.gettrace())
    width = 1280
    height = 720
    result_img = cv2.cvtColor(image__, cv2.COLOR_BGR2RGB)
    h, w, ch = result_img.shape
    bytes_per_line = ch * w
    qt_img = QImage(result_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    qt_img = QPixmap.fromImage(qt_img.scaled(width, height, Qt.KeepAspectRatio))
    return qt_img, values_


# Main
while True:
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())

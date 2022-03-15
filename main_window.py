import glob
import os
import random
import time

import torch
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import numpy as np
from PyQt5.QtWidgets import QMessageBox
from cv2 import cv2
from torch.backends import cudnn

from demo import Ui_Form
import sys

from utils.datasets import letterbox, vid_formats, img_formats, LoadImages, LoadStreams
from utils.general import check_img_size, check_imshow, scale_coords
from utils.torch_utils import select_device

from models.experimental import attempt_load
import os
from pathlib import Path


class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.label_image.setVisible(True)
        self.label_image.setScaledContents(True)

        test_img = QPixmap("test.jpg")
        self.label_image.setFixedSize(test_img.size())
        self.label_image.setPixmap(test_img)

        # 定时器
        self.timer_camera = QtCore.QTimer()
        self.timer_video = QtCore.QTimer()

        # 信号槽绑定
        self.pushButton_openimg.clicked.connect(self.open_img)
        self.pushButton_openvideo.clicked.connect(self.slot_video_button)
        self.pushButton_opencam.clicked.connect(self.slot_camera_button)
        self.pushButton_detect.clicked.connect(self.detect)

        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_video.timeout.connect(self.show_video)

        self.load_model()

    def load_model(self):
        # 加载模型
        root_path = os.getcwd()
        sys.path.insert(0, root_path + "/yolov5")
        self.device = select_device()
        self.model = attempt_load("mask_400_best.pt", map_location=self.device)

    ############################################################################################
    #
    #   打开图片
    #
    ############################################################################################

    def open_img(self):
        # 打开图片文件
        imgName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                  "All Files(*);;Text Files(*.txt)")

        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        print(names, colors)

        img = QPixmap(imgName)
        self.label_image.setFixedSize(img.size())
        self.label_image.setPixmap(img)

    ############################################################################################
    #
    #   处理视频逻辑
    #
    ############################################################################################
    def slot_video_button(self):
        # 视频
        if self.timer_video.isActive() == False:
            # 打开视频函数
            self.open_video()
        else:
            # 关闭视频函数
            self.close_video()

    def open_video(self):
        # 打开视频文件夹选取视频文件
        videoName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                    "All Files(*);")

        # 打开视频
        print(videoName)
        self.cap = cv2.VideoCapture(videoName)
        # self.cap.open(0)
        self.timer_video.start(30)
        self.pushButton_openvideo.setText("关闭视频")

    def close_video(self):
        # 关闭视频
        self.timer_video.stop()
        self.cap.release()
        self.label_image.clear()
        self.pushButton_openvideo.setText('打开视频')

    def show_video(self):
        # 显示视频图像
        flag, image = self.cap.read()
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.label_image.setFixedSize(showImage.size())
        self.label_image.setPixmap(QPixmap.fromImage(showImage))

    ############################################################################################
    #
    #   处理摄像头逻辑
    #
    ############################################################################################

    def slot_camera_button(self):
        if self.timer_camera.isActive() == False:
            print("打开摄像头")
            # 打开摄像头并显示图像信息
            self.open_camera()
        else:
            print("关闭摄像头")
            # 关闭摄像头并清空显示信息
            self.close_camera()

    def open_camera(self):
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        flag = self.cap.open(0)
        if flag is False:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                          buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

        else:
            self.timer_camera.start(30)
            self.pushButton_opencam.setText("关闭摄像头")

    def close_camera(self):
        # 关闭摄像头
        self.timer_camera.stop()
        self.cap.release()
        self.label_image.clear()
        self.pushButton_opencam.setText('打开摄像头')

    def show_camera(self):
        # 显示图像
        flag, image = self.cap.read()
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.label_image.setFixedSize(showImage.size())
        self.label_image.setPixmap(QPixmap.fromImage(showImage))

    ############################################################################################
    #
    #   检测
    #
    ############################################################################################

    def detect_img(self):
        model = self.model
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = [640, 640]  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        print(source)
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            source = str(source)
            device = select_device(self.device)
            webcam = False
            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            # Dataloader
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = 1  # batch_size
            vid_path, vid_writer = [None] * bs, [None] * bs
            # Run inference
            if pt and device.type != 'cpu':
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0
            for path, im, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1
                # Inference
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3
                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                # with open(txt_path + '.txt', 'a') as f:
                                #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                # if save_crop:
                                #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                #                  BGR=True)
                    # Print time (inference-only)
                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                    # Stream results
                    im0 = annotator.result()
                    # if view_img:
                    #     cv2.imshow(str(p), im0)
                    #     cv2.waitKey(1)  # 1 millisecond
                    # Save results (image with detections)
                    resize_scale = output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result.jpg", im0)
                    # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
                    self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))


if __name__ == "__main__":
    # app = QtWidgets.QApplication(sys.argv)
    # myshow = MainWindow()
    # myshow.show()
    # sys.exit(app.exec_())

    # root_path = os.getcwd()
    # sys.path.insert(0, root_path + "/yolov5")

    device = select_device()
    model = attempt_load("weights/mask_400_best.pt", map_location=device)
    stride = int(model.stride.max())  # model stride
    print("stride:", stride)
    imgsz = check_img_size(640, s=stride)  # check img_size
    print("image size:", imgsz)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    print("names:", names, "colors:", colors)

    dataset = LoadImages("yolov5/data/images", img_size=imgsz, stride=stride)
    for path, img, im0s, vid_cap in dataset:
        print(img)
        break

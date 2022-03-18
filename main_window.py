import threading

import torch
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import numpy as np
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem
from cv2 import cv2
from torch.backends import cudnn

# from demo import Ui_Form
from mask_img_detect import Ui_Form
import sys

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import os
from pathlib import Path
import inspect
import ctypes



def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


# 关闭线程
def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.init()

    def init(self):
        """初始化"""

        # 改变界面样式
        self.tabWidget.setTabText(0, "图片检测")
        self.tabWidget.setTabText(1, "视频检测")

        # 参数
        self.output_size = 480
        self.image = False
        self.video = False
        self.webcam = False

        self.label_image.setVisible(True)
        self.label_image.setScaledContents(True)

        # 初始界面图片
        self.img_source = "images/test_2.jpg"
        test_img = QPixmap(self.img_source)
        self.label_image.setFixedSize(test_img.size())
        self.label_image.setPixmap(test_img)

        # 定时器
        self.timer_camera = QtCore.QTimer()
        self.timer_video = QtCore.QTimer()

        # 信号槽绑定
        # 加载图片
        self.pushButton_openimg.clicked.connect(self.open_img)
        self.pushButton_detect.clicked.connect(self.detect_img)

        # 加载模型
        self.pushButton_choose_model.clicked.connect(self.load_model)
        self.pushButton_choose_model_2.clicked.connect(self.load_model)

        # 加载视频
        self.pushButton_openvideo.clicked.connect(self.slot_video_button)

        # 加载摄像头
        self.pushButton_opencamera.clicked.connect(self.slot_camera_button)


        # 初始化模型
        self.device = select_device()
        self.label_detect_device.setText(str(self.device))
        self.model = DetectMultiBackend("weights/best.pt", device=self.device, dnn=False)
        self.label_model.setText("weights/best.pt")

    ############################################################################################
    #
    #   模式切换
    #
    ############################################################################################

    def set_mode(self, mode):
        """切换模式
        mode=0:关闭所有
        mode=1:图片检测
        mode=2:视频检测
        mode=3:摄像头检测
        """
        self.image = False
        self.video = False
        self.webcam = False

        self.pushButton_choose_model_2.setEnabled(True)
        self.pushButton_opencamera.setEnabled(True)
        self.pushButton_openvideo.setEnabled(True)

        if mode == 0:  # 全部停止
            pass
        elif mode == 1:  # 检测图片
            self.image = True
        elif mode == 2:  # 检测视频
            self.video = True
            # 关闭摄像头功能
            self.pushButton_choose_model_2.setEnabled(False)
            self.pushButton_opencamera.setEnabled(False)
        elif mode == 3:  # 检测摄像头
            self.webcam = True
            # 关闭视频功能
            self.pushButton_choose_model_2.setEnabled(False)
            self.pushButton_openvideo.setEnabled(False)

    ############################################################################################
    #
    #   加载yolov5训练好的模型
    #
    ############################################################################################

    def load_model(self):
        modelName, modelType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.path.join(os.getcwd(), "weights"),
                                                                     "All Files(*);;*.pt *.pth")

        self.model = DetectMultiBackend(modelName, device=self.device, dnn=False)
        self.label_model.setText(modelName)
        print("模型加载完成")

    ############################################################################################
    #
    #   打开图片
    #
    ############################################################################################

    def open_img(self):
        # 打开图片文件
        imgName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.path.join(os.getcwd(), "images"),
                                                                  "All Files(*);;*.jpg *.png *.tif *.jpeg")
        print("打开图片：", imgName)
        self.img_source = imgName
        img = QPixmap(imgName)

        self.set_mode(1)
        self.label_image.setFixedSize(img.size())
        self.label_image.setPixmap(img)

    ############################################################################################
    #
    #   处理视频逻辑
    #
    ############################################################################################
    def slot_video_button(self):
        # 视频
        if not self.video:
            # 打开视频函数
            self.open_video()
        else:
            # 关闭视频函数
            self.close_video()

    def open_video(self):
        # 打开视频文件夹选取视频文件
        videoName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                    "All Files(*);;*.jpg *.png *.tif *.jpeg")

        # 打开视频
        self.vid_source = videoName

        # 视频检测
        self.set_mode(2)
        self.pushButton_openvideo.setText("关闭视频")

        self.th = threading.Thread(target=self.detect_img)
        self.th.start()

    def close_video(self):
        # 关闭视频
        self.set_mode(0)
        stop_thread(self.th)
        self.label_image_2.clear()
        self.pushButton_openvideo.setText('打开视频')

    ############################################################################################
    #
    #   处理摄像头逻辑
    #
    ############################################################################################

    def slot_camera_button(self):
        if not self.webcam:
            print("打开摄像头")
            # 打开摄像头并显示图像信息
            self.open_camera()
        else:
            print("关闭摄像头")
            # 关闭摄像头并清空显示信息
            self.close_camera()

    def open_camera(self):
        # 打开摄像头
        self.vid_source = '0'
        self.set_mode(3)
        self.pushButton_opencamera.setText('关闭摄像头')
        self.th = threading.Thread(target=self.detect_img)
        self.th.start()

    def close_camera(self):
        # 关闭摄像头
        self.set_mode(0)
        stop_thread(self.th)
        self.label_image_2.clear()
        self.pushButton_opencamera.setText('打开摄像头')

    ############################################################################################
    #
    #   开启检测
    #
    ############################################################################################

    def detect_img(self):
        device = self.device
        model = self.model
        output_size = self.output_size
        source = ''
        if self.image:
            source = self.img_source  # file/dir/URL/glob, 0 for webcam
        elif self.video or self.webcam:
            source = self.vid_source
            print("source:", source)
        imgsz = [640, 640]  # inference size (pixels)
        conf_thres = 0.5  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
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
        webcam = self.webcam
        tmp_dir = "images/tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        # print(f"开始检测图片：{source}")
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            source = str(source)
            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            # Dataloader
            if webcam:
                # view_img = check_imshow()
                # cudnn.benchmark = True  # set True to speed up constant image size inference
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
                        for _, c in enumerate(det[:, -1].unique()):
                            # 显示检测结果
                            n = (det[:, -1] == c).sum()  # detections per class
                            new_item = QTableWidgetItem(str(n.item()))
                            self.tableWidget_result.setItem(_, 0, new_item)
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
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond
                    # Save results (image with detections)
                    resize_scale = output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    tmp_imgpath = os.path.join(tmp_dir, "single_result.jpg")
                    cv2.imwrite(tmp_imgpath, im0)
                    img = QPixmap(tmp_imgpath)
                    # self.label_image.setFixedSize(img.size())
                    if self.image:
                        self.label_image.setPixmap(QPixmap(img))
                        self.label_detect_time.setText(f"{t3 - t2:.3f}s")
                    elif self.webcam or self.video:
                        self.label_image_2.setPixmap(QPixmap(img))
                        self.label_detect_time.setText(f"{t3 - t2:.3f}s")

            self.set_mode(0)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myshow = MainWindow()
    myshow.show()
    sys.exit(app.exec_())

import numpy as np
import cv2 as cv
import time
import sys,os

Conf_threshold = 0.1
NMS_threshold = 0.1
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
#class_name=['drone']

net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
#net = cv.dnn.readNetFromDarknet('yolov4-tiny_5.cfg', 'yolov4-tiny_5_final.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
model = cv.dnn_DetectionModel(net)
for img in os.listdir(os.getcwd()):
    if img.endswith("jpg"):
        frame=cv.imread(img)
        model.setInputParams(size=(960,736), scale=1/255, swapRB=True)
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        print(classes)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_name[classid], score)
            cv.rectangle(frame, box, color, 1)
            cv.putText(frame, label, (box[0], box[1]-10),cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        cv.imshow('frame', frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

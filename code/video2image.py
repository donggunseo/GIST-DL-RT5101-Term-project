import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

def video2img(filepath, second):
    video = cv2.VideoCapture(filepath)

    if not video.isOpened():
        print("Could not Open :", filepath)
        exit(0)


    #불러온 비디오 파일의 정보 출력
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(video.get(cv2.CAP_PROP_FPS))

    fps = fps * second

    count = 0
    image_list = []
    for count in tqdm(range(length)):
        ret, image = video.read()
        if(int(video.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 원하는 초(second)마다 추출
            image_list.append((count+1, image))
    video.release()

    return image_list

def detect(image, n):
    Width = image.shape[1]
    Height = image.shape[0]

    # read pre-trained model and config file
    net = cv2.dnn.readNet('../opencv_detection/yolov3.weights', '../opencv_detection/yolov3.cfg')

    # create input blob 
    # set input blob for the network
    net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))

    # run inference through the network
    # and gather predictions from output layers

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)


    class_ids = []
    confidences = []
    boxes = []

    #create bounding box 
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
    #check if is people detection
    for i in indices:
        box = boxes[i]
        if class_ids[i]==0: ## 
            y = round(box[1])
            h = round(box[3])
            x = round(box[0])
            w = round(box[2])
            if w > 200:
                a = int((860 - w) / 2)
                x1 = x-a if x-a>0 else 0
                x2 = x+w+a if x+w+a<Width else Width
                cropped_img = image[y : y + h, x1:x2]
                cv2.imwrite('../data/test_cropped/'+str(n)+'.jpg', cropped_img)
    return

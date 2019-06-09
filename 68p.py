# _*_ coding:utf-8 _*_

import os

import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

for img_path in os.listdir('imgs/'):
    if img_path.endswith(".txt"):
        continue
    # cv2读取图像
    img_path = "imgs/" + img_path
    if os.path.exists(img_path + '.txt'):
        continue
    img = cv2.imread(img_path)

    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 0)
    if not len(rects):
        print(img_path, "未识别到人脸,将移除{}".format(img_path))
        os.remove(img_path)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        with open(img_path + ".txt", 'w') as f:
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])
                f.write("{} {}\n".format(point[0, 0], point[0, 1]))

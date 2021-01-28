

#from app2 import app
import pandas as pd
import cv2
import numpy
#import requests
import math
import time
#from flask import render_template, request, flash, redirect
#import json
#import random
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
from head_pose_trial import get_2d_points, draw_annotation_box, head_pose_points
import logging
import time
import os
#import regex as re
#import base64
#from io import BytesIO
from PIL import Image
#import os.path
#from os import path
#from werkzeug.utils import secure_filename

import base64
import io
import cv2
from imageio import imread
#mport matplotlib.pyplot as plt

filename = "img1.jpg"

with open(filename, "rb") as fid:
    data = fid.read()

b64_bytes = base64.b64encode(data)





date_string = time.strftime("%Y-%m-%d-%H:%M")

def head_pose(b64):
    ts = time.strftime("%d-%b-%Y, %H:%M:%S")
    now = time.strftime("%d-%b-%Y")
    now2 = now + '.csv'
    name = os.path.join('./data', now2)
    file_status = str(os.path.exists(name))
    if file_status == 'False':
        # print("Not There")
        data_frame = {'input': [],'status': [], 'Time_stamp': []}
        df = pd.DataFrame(data_frame)
        df.to_csv(name, float_format='%.2f',
                  index=False, line_terminator=None)

    else:
        df = pd.read_csv(name)

    b64_string = b64.decode()
    image_data = b64
    im_name = b64[:10]
    # reconstruct image as an numpy array
    img = imread(io.BytesIO(base64.b64decode(b64_string)))

    logging.basicConfig(filename="Debug.log",
                      format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


    #image_data = re.sub('^data:image/.+;base64,', '', base64string)
    #im = Image.open(BytesIO(base64.b64decode(image_data)))
    #print(im)
    #filename = 'img.jpg'
    #im.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #logger.debug("base64 format decoded to image")
    path = './data'
    #img = cv2.imread(path)
    #logger.debug("reading completed")
    kwargs = {'a': image_data[-10:]}

    # img = cv2.imread(path)

    logger.debug("{a} ::: Reading Image ".format(**kwargs))

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    logger.debug("{a} ::: Reading and converting to RGB Completed".format(**kwargs))

    #rgb = img[100, 100]

    logger.debug("{a} ::: RGB Channels have been separated".format(**kwargs))

    #if rgb[0] > 2 and rgb[1] > 2 and rgb[2] > 2:
    logger.debug("{a} ::: Image is not Blocked".format(**kwargs))

    norm_img = numpy.zeros((300, 300))

    norm_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

    logger.debug("{a} ::: Image is Normalized".format(**kwargs))

    img = cv2.fastNlMeansDenoisingColored(norm_img, None, 3, 3, 5, 8)
    print('img:', img)
    logger.debug("{a} ::: Image is Denoising Completed".format(**kwargs))

    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    logging.debug('Calling get_face_detector and get_landmark_model')
    # cap = cv2.VideoCapture(0)
    #img = cv2.imread("C:/Users/Manoj/Desktop/up.jpg")
    # ret, img = cap.read()
    size = img.shape
    print(size)
    logger.debug("size of the image")
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 3D model points.
    model_points = numpy.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = numpy.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double"
    )
# while True:
# ret, img = cap.read()
# if ret == True:
    faces = find_faces(img, face_model)
    for face in faces:
        marks = detect_marks(img, landmark_model, face)
    # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
        image_points = numpy.array([
            marks[30],  # Nose tip
            marks[8],  # Chin
            marks[36],  # Left eye left corner
            marks[45],  # Right eye right corne
            marks[48],  # Left Mouth corner
            marks[54]  # Right mouth corner
            ], dtype="double")
        dist_coeffs = numpy.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

        (nose_end_point2D, jacobian) = cv2.projectPoints(numpy.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

        cv2.line(img, p1, p2, (0, 255, 255), 2)
        cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
        # for (x, y) in marks:
        #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
        # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
        try:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            ang1 = 90

        try:
            m = (x2[1] - x1[1]) / (x2[0] - x1[0])
            ang2 = int(math.degrees(math.atan(-1 / m)))
        except:
            ang2 = 90

        # print('div by zero error')
        #print('ang',ang1)
        #print('ang2', ang2)
        if ang1 >= 48:
            print('Head down')
            logger.debug('Head down')
            text='Head down'
            im = cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
            im = Image.fromarray(im)
            filename = '1.jpg'
            im.save(os.path.join('./data', filename))
            rest=[{'input':im_name,'status':text,'time_stamp':ts}]
            print(rest)
            df.loc[len(df.index)] = list(rest[0].values())
            df.to_csv(name, index=False)
            return "Head Down"
        elif ang1 <= -48:
            print('Head up')
            logger.debug('Head up')
            text='head up'
            im = cv2.putText(img, 'Head up', (15, 15), font, 2, (255, 255, 128), 3)
            im = Image.fromarray(im)
            filename = '1.jpg'
            im.save(os.path.join('./data', filename))
            rest=[{'input':im_name,'status':text,'time_stamp':ts}]
            print(rest)
            df.loc[len(df.index)] = list(rest[0].values())
            df.to_csv(name, index=False)
            return "Head up"
        if ang2 >= 40:
            print('Head right')
            logger.debug('Head right')
            text='Head right'
            im = cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
            im = Image.fromarray(im)
            filename = '1.jpg'
            im.save(os.path.join('./data', filename))
            rest=[{'input':im_name,'status':text,'time_stamp':ts}]
            print(rest)
            df.loc[len(df.index)] = list(rest[0].values())
            df.to_csv(name, index=False)
            return "Head Right"
        elif ang2 <= -48:
            print('Head left')
            logger.debug('Head left')
            text='Head left'
            im = cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
            im = Image.fromarray(im)
            filename = '1.jpg'
            im.save(os.path.join('./data', filename))
            rest=[{'input':im_name,'status':text,'time_stamp':ts}]
            print(rest)
            df.loc[len(df.index)] = list(rest[0].values())
            df.to_csv(name, index=False)
            return "Head Left"

            cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
    s = cv2.imshow('img', img)
    return s
    cv2.waitKey(0)
    cv2.destroyAllWindows()
head_pose(b64_bytes)
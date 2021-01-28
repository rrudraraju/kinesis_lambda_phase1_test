
#from app2 import app
#import face_recognition
import cv2
import numpy as np
#from flask import render_template, request, flash, redirect
import logging
import time
import os
#import regex as re
import base64
#from io import BytesIO
#from PIL import Image
import os.path
#import math
import pandas as pd
from PIL import Image

filepath='./data'




#import base64
import io
#import cv2
from imageio import imread
#import matplotlib.pyplot as plt

filename = "img.jpg"
with open(filename, "rb") as fid:
    data = fid.read()

b64_bytes = base64.b64encode(data)
#b64_string = b64_bytes.decode()

# reconstruct image as an numpy array
#img = imread(io.BytesIO(base64.b64decode(b64_string)))


def face_detect(b64):
    logging.basicConfig(filename="Debug.log",format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    image_data = b64
    ts = time.strftime("%d-%b-%Y, %H:%M:%S")

    #image_data = re.sub('^data:image/.+;base64,', '', base64string)
    b64_string = b64.decode()
    #im = Image.open(BytesIO(base64.b64decode(image_data)))
    img = imread(io.BytesIO(base64.b64decode(b64_string)))
    #print(im)
    #filename = 'img.jpg'
    #im.save(os.path.join(filepath, filename))
    #logger.debug("base64 format decoded to image")
    path = './data/'
    #img = cv2.imread(path)
    #logger.debug("reading completed")
    #img = cv2.resize(img, (600, 400))
    now = time.strftime("%d-%b-%Y")
    now2 = now + '.csv'
    name = os.path.join('data/', now2)
    file_status = str(os.path.isfile(name))
    if file_status == 'False':
        # print("Not There")
        data_frame = {'input': [],'status': [], 'Time_stamp': []}
        df = pd.DataFrame(data_frame)
        df.to_csv(name, float_format='%.2f',
                  index=False, line_terminator=None)
    else:
        df = pd.read_csv(name)
        #size1 = img.shape
        #print(size1)
        #h, w, c = img.shape

        #if h == 400 and w == 600 and c == 3:

        kwargs = {'a': image_data[-10:]}

        # img = cv2.imread(path)

        logger.debug("{a} ::: Reading Image ".format(**kwargs))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        logger.debug("{a} ::: Reading and converting to RGB Completed".format(**kwargs))

        rgb = img[100, 100]

        logger.debug(

                "{a} ::: RGB Channels have been separated".format(**kwargs))

        if rgb[0] > 2 and rgb[1] > 2 and rgb[2] > 2:
            logger.debug("{a} ::: Image is not Blocked".format(**kwargs))

            norm_img = np.zeros((300, 300))

            norm_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

            logger.debug("{a} ::: Image is Normalized".format(**kwargs))

            img = cv2.fastNlMeansDenoisingColored(norm_img, None, 3, 3, 5, 8)
            logger.debug("{a} ::: Image is Denoising Completed".format(**kwargs))

            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            #img = cv2.imread(imagePath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            eye = eye_cascade.detectMultiScale(gray, 1.1, 5)

            if len(faces) > 0:

                if len(eye) > 0:
                    logger.debug("{a} ::: Face Detected".format(**kwargs))

                    #result = {"label": "Attentive", "score": "100%"}
                    text='Face Detected and eyes detected'
                    #im = cv2.putText(img, text, (left, top - 50), #cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                    #im = Image.fromarray(im)
                    #filename = 'mo'+now+'.jpg'
                    #im.save(os.path.join(filepath, filename))
                    rest = [{'input':image_data[0:5], 'status': text, 'Time_stamp':ts}]
                    print(rest)
                    df.loc[len(df.index)] = list(rest[0].values())
                    df.to_csv(name, index=False)

                    return 'Face Detected and eyes detected'
                else:

                    logger.debug("{a} ::: Face Detected Eye's not detected".format(**kwargs))

                    #result = {"label": "Not So Attentive", "score": "50%"}
                    text='Face Detected Eyes not detected'
                    #im = cv2.putText(img, text, (left, top - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                    #im = Image.fromarray(im)
                   # filename = 'mo'+now+'.jpg'
                    #im.save(os.path.join(filepath, filename))
                    rest = [{'input':image_data[0:5], 'status': text, 'Time_stamp':ts}]
                    print(rest)
                    df.loc[len(df.index)] = list(rest[0].values())
                    df.to_csv(name, index=False)

                    return 'Face Detected Eyes not detected'

            else:

                logger.debug("{a} ::: No Face detected".format(**kwargs))

                #result = {"label": "Not Attentive", "score": "0%"}
                text='Face not detected'
                #im = cv2.putText(img, text, (left, top - 50), #cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                #im = Image.fromarray(im)
                #filename = 'mo'+now+'.jpg'
                #im.save(os.path.join(filepath, filename))
                rest = [{'input':image_data[0:5], 'status': text, 'Time_stamp':ts}]
                print(rest)
                df.loc[len(df.index)] = list(rest[0].values())
                df.to_csv(name, index=False)

                return 'Face not detected'

        else:

            logger.debug("{a} ::: can't read image ".format(**kwargs))

            #result = {"label": "No face Detected", "score": "null"}
            text='No face Detected'
            #im = cv2.putText(img, text, (left, top - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            #im = Image.fromarray(im)
            #filename = 'mo'+now+'.jpg'
            #im.save(os.path.join(filepath, filename))
            rest = [{'input':image_data[0:5], 'status': text, 'Time_stamp':ts}]
            print(rest)
            df.loc[len(df.index)] = list(rest[0].values())
            df.to_csv(name, index=False)

            return 'No face Detected'

        # else:

        # logger.debug("{a} ::: Not in format".format(**kwargs))

        # result = {'Attenton': 'this size not accepted'}

        # return redirect('/')

        # except:
        #
        # logger.critical("Error Caught", exc_info=True)
face_detect(b64_bytes)

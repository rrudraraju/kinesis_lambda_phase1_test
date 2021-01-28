import face_recognition
import cv2
import numpy as np
from mouth_open_algorithm import get_lip_height, get_mouth_height
import logging
import os
import regex as re
import base64
from io import BytesIO
from PIL import Image
import os.path
import math
import time
import io
from imageio import imread
import pandas as pd

data_string = time.strftime("%Y-%m-%d")
path = './data/'
filename = "img1.jpg"
with open(filename, "rb") as fid:
    data = fid.read()

b64_bytes = base64.b64encode(data)
print(b64_bytes)


def is_mouth_open(face_landmarks):
    top_lip = face_landmarks['top_lip']
    bottom_lip = face_landmarks['bottom_lip']

    top_lip_height = get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)

    # if mouth is open more than lip height * ratio, return true.
    ratio = 0.5
    print('top_lip_height: %.2f, bottom_lip_height: %.2f, mouth_height: %.2f, min*ratio: %.2f'
          % (top_lip_height, bottom_lip_height, mouth_height, min(top_lip_height, bottom_lip_height) * ratio))

    if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
        return True
    else:
        return False


def mouth_detect(b64):

    now = time.strftime("%d-%b-%Y")
    ts = time.strftime("%d-%b-%Y, %H:%M:%S")
    now2 = now + '.csv'
    name = os.path.join(path, now2)
    file_status = str(os.path.exists(name))
    kwargs = {'a': b64_bytes[-10:]}
    if file_status == 'False':
        # print("Not There")
        data_frame = {'input': [], 'status': [], 'Time_stamp': []}
        df = pd.DataFrame(data_frame)
        df.to_csv(name, float_format='%.2f',
                  index=False, line_terminator=None)

    else:
        df = pd.read_csv(name)
        b64_string = b64.decode()
        im_name = b64[:10]

        image = imread(io.BytesIO(base64.b64decode(b64_string)))
        kwargs = {'a': b64_bytes[-10:]}
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # logger.debug("{a} ::: Reading and converting to RGB Completed".format(**kwargs))

        rgb = img[100, 100]

        # logger.debug(
        #
        #     "{a} ::: RGB Channels have been separated".format(**kwargs))

        if rgb[0] > 2 and rgb[1] > 2 and rgb[2] > 2:
            # logger.debug("{a} ::: Image is not Blocked".format(**kwargs))

            norm_img = np.zeros((300, 300))

            norm_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

            # logger.debug("{a} ::: Image is Normalized".format(**kwargs))

            img = cv2.fastNlMeansDenoisingColored(norm_img, None, 3, 3, 5, 8)

        while True:
            # Grab a single frame of video
            # ret, frame = video_capture.read()
            # Find all the faces and face encodings in the frame of video
            face_locations = face_recognition.face_locations(img)
            face_encodings = face_recognition.face_encodings(img, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(img)

            # Loop through each face in this frame of video
            for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings,
                                                                                 face_landmarks_list):

                # Draw a box around the face
                cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(img, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(frame, name, (left + 6, bottom + 25), font, 1.0, (255, 255, 255), 1)

                # Display text for mouth open / close
                ret_mouth_open = is_mouth_open(face_landmarks)
                if ret_mouth_open is True:
                    text = 'Mouth is open'
                    print('Mouth is open')
                    logging.debug('Mouth is open')
                    im = cv2.putText(img, text, (left, top - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                    im = Image.fromarray(im)
                    filename = 'mo' + data_string + '.jpg'
                    im.save(os.path.join(path, filename))
                    rest = [{'input': im_name, 'status': 'Mouth_open', 'Time_stamp': ts}]
                    print(rest)
                    df.loc[len(df.index)] = list(rest[0].values())
                    df.to_csv(name, index=False)
                    return 'Mouth is Open'
                else:
                    text = 'not open'
                    print('not open')
                    logging.debug('Mouth Not open')
                    im = cv2.putText(img, text, (left, top - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                    im = Image.fromarray(im)
                    filename = 'no' + data_string + '.jpg'
                    im.save(os.path.join(path, filename))
                    rest = [{'input': im_name, 'status': 'Mouth_not_open', 'Time_stamp': ts}]
                    print(rest)
                    df.loc[len(df.index)] = list(rest[0].values())
                    df.to_csv(name, index=False)
                    return 'Not Open'
                # im = cv2.putText(img, text, (left, top - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                # im = Image.fromarray(im)
                # filename = 'mo.jpg'
                # im.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Display the resulting image
            cv2.imshow('Image', img)
            # out.write(frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        # video_capture.release()
        cv2.destroyAllWindows()

mouth_detect(b64_bytes)









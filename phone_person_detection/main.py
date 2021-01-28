




from secondary import load_darknet_weights,draw_outputs,DarknetConv,DarknetResidual,DarknetBlock,Darknet,YoloConv,YoloOutput,yolo_boxes,yolo_nms,YoloV3,weights_download
import base64
import io ,time,os
import cv2
import pandas as pd
import numpy as np
from imageio import imread
from PIL import Image
#import matplotlib.pyplot as plt

filename = "img1.jpg"
with open(filename, "rb") as fid:
    data = fid.read()

b64_bytes = base64.b64encode(data)


# reconstruct image as an numpy array

path='./data'


def ph_pr(b64):
    now = time.strftime("%d-%b-%Y")
    ts = time.strftime("%d-%b-%Y, %H:%M:%S")
    now2 = now + '.csv'
    name = os.path.join(path, now2)
    file_status = str(os.path.exists(name))
    if file_status == 'False':
        # print("Not There")
        data_frame = {'input': [],'status': [], 'time_stamp': []}
        df = pd.DataFrame(data_frame)
        df.to_csv(name, float_format='%.2f',index=False, line_terminator=None)

    else:
        df = pd.read_csv(name)

    b64_string = b64.decode()
    im_name = b64[:10]

    image = imread(io.BytesIO(base64.b64decode(b64_string)))
    #image = cv2.imread('img2.jpg')
    #while (True):
    #    ret, image = cap.read()
    #    if ret == False:
    #        break
    yolo = YoloV3()
    load_darknet_weights(yolo, 'yolov3.weights')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    img = img / 255
    class_names = [c.strip() for c in open("classes.TXT").readlines()]
    boxes, scores, classes, nums = yolo(img)
    count = 0
    for i in range(nums[0]):
        if int(classes[0][i] == 0):
            count += 1
        if int(classes[0][i] == 67):
            print('Mobile Phone detected')
            text='mobile phone detected'
            #im=cv2.putText(img,text,(30,30),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)
            #im = Image.fromarray(im)
            #im=cv2.imwrite(path,im)
            #file="./data/text+'.jpg'"
            #im=cv2.imwrite(filename,im)
            #im.save(os.path.join(path,filename))
            rest=[{'input':im_name,'status':text,'time_stamp':ts}]
            print(rest)
            df.loc[len(df.index)] = list(rest[0].values())
            df.to_csv(name, index=False)
    if count == 0:
        print('No person detected')
        text='No person detected'
        #im=cv2.putText(img,text,(30,30),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)
        #im = Image.fromarray(im)
        
        #file="./data/text+'.jpg'"
        #im=cv2.imwrite(filename,im)
        #im.save(os.path.join(path,filename))
        rest=[{'input':im_name,'status':text,'time_stamp':ts}]
        print(rest)
        df.loc[len(df.index)] = list(rest[0].values())
        df.to_csv(name, index=False)
    elif count > 1:
        print('More than one person detected')
        text='More than one person detected'
       # im =cv2.putText(img,text,(30,30),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)
        #im = Image.fromarray(im)
        #file="./data/text+'.jpg'"
        #im=cv2.imwrite(filename,im)
        rest=[{'input': im_name,'status':text,'time_stamp':ts}]
        print(rest)
        df.loc[len(df.index)] = list(rest[0].values())
        df.to_csv(name, index=False)

    image = draw_outputs(image, (boxes, scores, classes, nums), class_names)


ph_pr(b64_bytes)

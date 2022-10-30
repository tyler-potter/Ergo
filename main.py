import cv2
import time

import os

thres = 0.5

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = 'models/coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'models/frozen_inference_graph.pb'


net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

confidenceScore = 0 # if value is above 0.7, active punishment

efficencyMetric = 0

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)

    if 77 in classIds:
        print("OFF TASK    !!!   Cell Phone Detected")
    else:
        print("ON TASK           No Cell Phone Detected")

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output",img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
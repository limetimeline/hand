import cv2
from cvzone.HandTrackingModule import HandDetector
from cvfpscalc import CvFpsCalc

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

import socket

cvFpsCalc = CvFpsCalc(buffer_len=10)


def Calc_distance(pointLeft, pointRight):
    cv2.circle(img,pointLeft, 5, (255,0,255), cv2.FILLED)
    cv2.circle(img,pointRight, 5, (255,0,255), cv2.FILLED)
    cv2.line(img,pointLeft,pointRight,(0,200,0),3)
    
    w,_ = detector.findDistance(pointLeft, pointRight)
    W = 6.3
    
    # Finding the Focal Length
    d = 50
    # f = (w*d)/W
    # print(f)
    
    # Finding distance
    f = 1000
    d = (W*f)/w
    return d


 
# Parameters
width, height = 1280, 720

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detector = HandDetector(maxHands = 2, detectionCon = 0.8)

# Communication
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

while True:
    # Get the frame from the webcam
    success, img = cap.read()
    #Hands
    hands, img = detector.findHands(img)
    
    fps = cvFpsCalc.get()
    strfps = "[GPU] FPS : %0.1f" % float(fps)
    img = cv2.putText(img, strfps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4, cv2.LINE_AA)
    
    data = []
   
    # Landmark values - (x,y,z) * 21 [총 21개.(63개)]
    if hands:
        # print('\n')
        # Get the first hand detected
        hand = hands[0]
        # Get the landmark list
        lmList = hand['lmList']
        
        d = Calc_distance([lmList[5][0], lmList[5][1]], [lmList[17][0], lmList[17][1]])
        
        
        
        xs = []
        ys = []
        zs = []
        
        # print(lmList)
        for lm in lmList:
            xs.append(lm[0])
            ys.append(height - lm[1])
            zs.append((lm[2]+80) + d)
            # data.extend([lm[0], height - lm[1], lm[2]])
            data.extend([lm[0], height - lm[1], (lm[2]+80) + d])
            
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.scatter(xs, ys, zs, c='darkgreen', marker='o', s=15)
        
        print(data)
        sock.sendto(str.encode(str(data)), serverAddressPort)
    
         
    img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    # plt.show()
    cv2.waitKey(1)
    

    
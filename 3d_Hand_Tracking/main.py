import cv2
from cvzone.HandTrackingModule import HandDetector
from cvfpscalc import CvFpsCalc

import socket

cvFpsCalc = CvFpsCalc(buffer_len=10)

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
    img = cv2.putText(img, strfps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    
    data = []
    # Landmark values - (x,y,z) * 21 [총 21개.(63개)]
    if hands:
        # print('\n')
        # Get the first hand detected
        hand = hands[0]
        # Get the landmark list
        lmList = hand['lmList']
        # print(lmList)
        for lm in lmList:
            data.extend([lm[0], height - lm[1], lm[2]])
        # print(data)
        sock.sendto(str.encode(str(data)), serverAddressPort)
        
    img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
    
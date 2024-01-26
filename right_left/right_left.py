import cv2
import mediapipe as mp
import numpy as np
from tensorflow.python.client import device_lib
# from utils import CvFpsCalc
from cvfpscalc import CvFpsCalc
import os
# CPU 강제 사용을 원하신다면 -1로 번호를 선택해주시면 됩니다.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# print(device_lib.list_local_devices())

cvFpsCalc = CvFpsCalc(buffer_len=10)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    use_brect = True
    gesture = [0 for _ in range(5)]
    inf_hand = bin(192) #0b11000000, 앞의 11은 아무의미없음

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image_height, image_width, _ = image.shape
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        word = []
        handType = ""

        if results.multi_hand_landmarks:
            for hand, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                handType = str(hand.classification[0].label)

                # # 좌표 확인
                # for ids, landmrk in enumerate(hand_landmarks.landmark):
                #     cx, cy = landmrk.x * image_width, landmrk.y * image_height
                #     print(f"<{ids}>\nx : {cx}\ny: {cy} \nz:{landmrk.z}")
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style())

                thumbIsOpen = False
                firstFingerIsOpen = False
                secondFingerIsOpen = False
                thirdFingerIsOpen = False
                fourthFingerIsOpen = False

                pseudoFixKeyPoint = hand_landmarks.landmark[4].y
                if (hand_landmarks.landmark[5].y > pseudoFixKeyPoint) :
                    thumbIsOpen = True
                    inf_hand = bin(int(inf_hand,2) | int(bin(16),2))
                else:
                    inf_hand = bin(int(inf_hand,2) & ~int(bin(16),2))

                pseudoFixKeyPoint = hand_landmarks.landmark[6].y
                if (hand_landmarks.landmark[7].y < pseudoFixKeyPoint) and (
                        hand_landmarks.landmark[8].y < pseudoFixKeyPoint):
                    firstFingerIsOpen = True
                    inf_hand = bin(int(inf_hand,2) | int(bin(8),2))
                else:
                    inf_hand = bin(int(inf_hand,2) & ~int(bin(8),2))

                pseudoFixKeyPoint = hand_landmarks.landmark[10].y
                if (hand_landmarks.landmark[11].y < pseudoFixKeyPoint) and (
                        hand_landmarks.landmark[12].y < pseudoFixKeyPoint):
                    secondFingerIsOpen = True
                    inf_hand = bin(int(inf_hand,2) | int(bin(4),2))
                else:
                    inf_hand = bin(int(inf_hand,2) & ~int(bin(4),2))

                pseudoFixKeyPoint = hand_landmarks.landmark[14].y
                if (hand_landmarks.landmark[15].y < pseudoFixKeyPoint) and (
                        hand_landmarks.landmark[16].y < pseudoFixKeyPoint):
                    thirdFingerIsOpen = True
                    inf_hand = bin(int(inf_hand,2) | int(bin(2),2))
                else:
                    inf_hand = bin(int(inf_hand,2) & ~int(bin(2),2))

                pseudoFixKeyPoint = hand_landmarks.landmark[18].y
                if (hand_landmarks.landmark[19].y < pseudoFixKeyPoint) and (
                        hand_landmarks.landmark[20].y < pseudoFixKeyPoint):
                    fourthFingerIsOpen = True
                    inf_hand = bin(int(inf_hand,2) | int(bin(1),2))
                else:
                    inf_hand = bin(int(inf_hand,2) & ~int(bin(1),2))

                if handType == 'Right':
                    inf_hand = bin(int(inf_hand,2) & ~int(bin(32),2))
                else:
                    inf_hand = bin(int(inf_hand, 2) | int(bin(32), 2))

                word.append(inf_hand)

                # if (thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen) :
                #     word = "FIVE"
                # elif (not thumbIsOpen) and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen :
                #     word = "FOUR"
                # elif thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and (not thirdFingerIsOpen) and (not fourthFingerIsOpen) :
                #     word = "THREE"
                # elif thumbIsOpen and firstFingerIsOpen and (not secondFingerIsOpen) and (not thirdFingerIsOpen) and (not fourthFingerIsOpen) :
                #     word = "TWO"
                # elif (not thumbIsOpen) and firstFingerIsOpen and (not secondFingerIsOpen )and (not thirdFingerIsOpen) and (not fourthFingerIsOpen) :
                #     word = "ONE"
                # elif (not thumbIsOpen) and firstFingerIsOpen and secondFingerIsOpen and (not thirdFingerIsOpen) and (not fourthFingerIsOpen) :
                #     word = "YEAH"
                # elif not thumbIsOpen and firstFingerIsOpen and not secondFingerIsOpen and not thirdFingerIsOpen and fourthFingerIsOpen :
                #     word = "ROCK!"
                # elif thumbIsOpen and firstFingerIsOpen and not secondFingerIsOpen and not thirdFingerIsOpen and fourthFingerIsOpen :
                #     word = "SPIDERMAN!"
                # elif not (thumbIsOpen or firstFingerIsOpen or secondFingerIsOpen or thirdFingerIsOpen or fourthFingerIsOpen) :
                #     word = "FIST"
        fps = cvFpsCalc.get()
        strfps = "[GPU] FPS : %0.1f" % float(fps)

        image = cv2.flip(image, 1)
        image = cv2.putText(image, strfps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        for i in range(len(word)):
            if word[i][4] == '0' :  #오른손
                image = cv2.putText(image, word[i][4:], (300, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3,
                                    cv2.LINE_AA)
            else :
                image = cv2.putText(image, word[i][4:], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3,
                                    cv2.LINE_AA)
        cv2.imshow('MediaPipe Hands', image)

        # 27 > ESC 키
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
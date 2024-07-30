# -*- coding: utf-8 -*-
import cv2
import numpy as np
import serial
import math
import time

ser = serial.Serial('/dev/ttyTHS1', 115200) 

ratio = 0.6184
CenterX = 175
CenterY = 135
First1=200
First2=120
CapX=320
CapY=240

def location(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    edge1 = np.linalg.norm(box[0] - box[1])
    edge2 = np.linalg.norm(box[1] - box[2])
    edge=round((edge1+edge2)/2)
    # print(edge)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
    for point in box:
        cv2.circle(img, tuple(point), 5, (0, 255, 0), -1)
        cv2.putText(img, f'({point[0]},{point[1]})', point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    width = int(rect[1][0])
    height = int(rect[1][1])

    warped = gray

    circles = cv2.HoughCircles(warped, method=cv2.HOUGH_GRADIENT,
                               dp=1, minDist=25, param1=100, param2=19,
                               minRadius=10, maxRadius=20)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        centers = []

        grid_size1 = width // 3
        grid_size2 = height // 3
        

        for i in circles[0, :]:
            cx = int(i[0])
            cy = int(i[1])
            r = i[2]
            row = round(cy / grid_size1)
            col = round(cx / grid_size2)
            cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)
            centers.append((cx, cy, row, col))
    else:
        centers = []


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)


    res_black = cv2.bitwise_and(img, img, mask=mask_black)
    res_white = cv2.bitwise_and(img, img, mask=mask_white)

   
    res_black = cv2.cvtColor(res_black, cv2.COLOR_BGR2GRAY)
    res_white = cv2.cvtColor(res_white, cv2.COLOR_BGR2GRAY)

    stones = []
    white = []
    black = []
    Black=[]
    White=[]
    for center in centers:
        cx, cy, row, col = center
        if ((row>=1 and row<=3) and (col>=1 and col<=3)):
            black_roi = res_black[cy - r:cy + r, cx - r:cx + r]
            nz_count_black = cv2.countNonZero(black_roi)

            if nz_count_black > 700:
                print(nz_count_black)
                color = 'black'
                stones.append((color, row, col, cx, cy))
                cx = cx + CenterX
                cy = cy + CenterY
                cx1 = int(cx / 100)
                cx2 = cx % 100
                cy1 = int(cy / 100)
                cy2 = cy % 100
                black.append((cx1, cx2, cy1, cy2))
                Black.append(row)
                Black.append(col)
                print(str('0x1d'),row,col)
                cv2.putText(img, f'({cx},{cy})', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            if nz_count_black < 100:
                print(nz_count_black)
                color = 'white'
                stones.append((color, row, col, cx, cy))
                cx = cx + CenterX
                cy = cy + CenterY
                cx1 = int(cx / 100)
                cx2 = cx % 100
                cy1 = int(cy / 100)
                cy2 = cy % 100
                white.append((cx1, cx2, cy1, cy2))
                White.append(row)
                White.append(col)
                print(str('0x2d'),row,col)
                cv2.putText(img, f'({cx},{cy})', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                continue
        else:
            print(str('0x3d'),row,col)
            continue
    board = [[0 for _ in range(10)] for _ in range(10)]

    for stone in stones:
        color, row, col, cx, cy = stone

        board[row][col] = 1 if color == 'white' else 2
    # Box = [(b[0] + CenterX, b[1] + CenterY) for b in box]
    cv2.imshow('img',img)
    cv2.waitKey(1)
    Box=(box+[First2,First1]-[CapX,CapY])*[ratio,-ratio]+[CenterX+CenterY]

    return Box, Black, White

def send_data_ifNumber2(array):
    ser.write(chr(0x77).encode())
    ser.write(chr(0x77).encode())
    ser.write(chr(len(array) + 4).encode())
    ser.write(chr(0x1d).encode())
    for i in range(len(array)):
        ser.write(chr(array[i]).encode())
    ser.write(chr(0x5B).encode())
def send_data_ifNumber3(array1, array2):
    number1 = len(array1)
    number2 = len(array2)
    array = array1 + array2
    ser.write(chr(0x77).encode())
    ser.write(chr(0x77).encode())
    ser.write(chr(len(array) + 6).encode())
    ser.write(chr(0x2d).encode())
    ser.write(chr(number1//2).encode())
    for i in range(number1):
        ser.write(chr(array1[i]).encode())
    ser.write(chr(number2//2).encode())
    for j in range(number2):
        ser.write(chr(array2[j]).encode())
    ser.write(chr(0x5B).encode())

receive = [0, 0, 0, 0, 0]
Number = 1
send_data1 = []

while True:
    print(Number)
    if Number == 1:
        try:
            receive[0] = ser.read()
            receive[0] = hex(ord(receive[0]))
            while receive[0] != '0x77':
                receive[0] = ser.read()
            receive[1] = ser.read()
            receive[2] = ser.read()
            receive[3] = ser.read()
            receive[4] = ser.read()
            receive[1] = hex(ord(receive[1]))
            receive[2] = hex(ord(receive[2]))
            receive[3] = hex(ord(receive[3]))
            receive[4] = hex(ord(receive[4]))
            # print(receive)
            if receive[3] != 0:
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                frame=frame[120:350,200:422]
                if ret:
                    if receive[3] == '0x1f':
                        Number = 2
                    elif receive[3] == '0x2f':
                        Number = 3

        except serial.SerialException as e:
            print("ERROR1:", e)
        except Exception as e:
            print("ERROR2:", e)

    elif Number == 2:
        box, _, _= location(frame)
        send_data1 = []
        x1, y1 = box[0][0] , box[0][1]
        x2, y2 = box[1][0] , box[1][1]
        x3, y3 = box[2][0] , box[2][1]
        x4, y4 = box[3][0] , box[3][1]
        edge1=math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
        edge2=math.sqrt((x2-x3)*(x2-x3)+(y2-y3)*(y2-y3))
        edge=round((edge1+edge2)/2)
        # print(edge)
        # print(x1)
        # print(y1)
        # print(x2)
        # print(y2)
        # print(x3)
        # print(y3)
        # print(x4)
        # print(y4)
        send_data1.append(int(x1 // 100))
        send_data1.append(round(x1 % 100))
        send_data1.append(int(y1 // 100))
        send_data1.append(round(y1 % 100))
        send_data1.append(int(x2 // 100))
        send_data1.append(round(x2 % 100))
        send_data1.append(int(y2 // 100))
        send_data1.append(round(y2 % 100))
        send_data1.append(int(x3 // 100))
        send_data1.append(round(x3 % 100))
        send_data1.append(int(y3 // 100))
        send_data1.append(round(y3 % 100))
        send_data1.append(int(x4 // 100))
        send_data1.append(round(x4 % 100))
        send_data1.append(int(y4 // 100))
        send_data1.append(round(y4 % 100))
        # print(send_data1)
        send_data_ifNumber2(send_data1)
        Number = 1

    elif Number == 3:
        _, Black, White= location(frame)
        print(Black)
        print(White)
        send_data_ifNumber3(Black, White)
        Number = 1
    cap.release()
    cv2.destroyAllWindows()
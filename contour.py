import numpy as np
import cv2
import sys
from decimal import Decimal

device =cv2.VideoCapture(0)
while True:

    _, frame = device.read()
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    thresh = cv2.threshold(blurred_frame, 1, 255, cv2.THRESH_BINARY)[1]
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    lower_red=np.array([166,84,141])
    upper_red=np.array([186,255,255])
     
    mask= cv2.inRange(hsv, lower_red,upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    res=cv2.bitwise_and(blurred_frame, frame, mask=mask)

    for contour in contours:

        area = cv2.contourArea(contour)
        epsilon = 0.1*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)     

        if area > 500:

            cv2.drawContours(blurred_frame, contour, -1, (0, 255, 0), 3)
            cv2.drawContours(res, [approx], -1, (255, 0, 0), 3)

    M=cv2.moments(mask)

    if M["m00"]!=0:

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    else:

        cX = int(M["m10"])
        cY = int(M["m01"])

    cv2.circle(mask, (cX,cY), 5, (0,255,255), -1)

    cv2.imshow('frame',blurred_frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k=cv2.waitKey(1) & 0xFF
    if k==27:
    	break

device.release()
cv2.destroyAllWindows()
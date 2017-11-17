# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:04:35 2017

@author: veerachart
"""

import cv2
import numpy as np
import time

angles = range(0,360,4)
window_sizes = [(32,64), (40,80), (56,112), (60,120)]
#window_sizes = [(64,128)]
winStride = (5,5)#(10,10)#

img = cv2.imread('../FisheyeHOG/image/pedestriansimulated.jpg')
#img = cv2.resize(img,(384,384))
image = img.copy()
rows, cols, channels = img.shape

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

detections = []

t0 = time.time()
for angle in angles:
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols,rows))
    #cv2.imshow("Rotated",dst)
    #cv2.waitKey(100)
    for window in window_sizes:
        #t0 = time.time()
#        hog = cv2.HOGDescriptor(window,(16,16),(8,8),(8,8),9)
#        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        width = window[0]
        height = window[1]
        roi = dst[0:rows/2, (cols-width)/2:(cols+width)/2]
        scale = 64./width
        roi = cv2.resize(roi,(64,int(rows*scale/2)))
        (detects, weights) = hog.detect(roi, winStride=winStride, padding=(0,0))
        #t1 = time.time()
        #print t1-t0
        if len(detects):
            for (x,y) in detects:
                angle_rad = angle*np.pi/180.
                u1 = int(-np.cos(angle_rad)*width/2 + np.sin(angle_rad)*(rows/2-y/scale) + cols/2)
                v1 = int(-np.sin(angle_rad)*width/2 - np.cos(angle_rad)*(rows/2-y/scale) + rows/2)
                u2 = int(np.cos(angle_rad)*width/2 + np.sin(angle_rad)*(rows/2-y/scale) + cols/2)
                v2 = int(np.sin(angle_rad)*width/2 - np.cos(angle_rad)*(rows/2-y/scale) + rows/2)
                u3 = int(np.cos(angle_rad)*width/2 + np.sin(angle_rad)*(rows/2-y/scale-height) + cols/2)
                v3 = int(np.sin(angle_rad)*width/2 - np.cos(angle_rad)*(rows/2-y/scale-height) + rows/2)
                u4 = int(-np.cos(angle_rad)*width/2 + np.sin(angle_rad)*(rows/2-y/scale-height) + cols/2)
                v4 = int(-np.sin(angle_rad)*width/2 - np.cos(angle_rad)*(rows/2-y/scale-height) + rows/2)
                detections.append([(u1,v1),(u2,v2),(u3,v3),(u4,v4)])
t1 = time.time()
print t1-t0
if len(detections):
    for detection in detections:
        cv2.line(img,detection[0],detection[1],(0,0,255),2)
        cv2.line(img,detection[1],detection[2],(0,0,255),2)
        cv2.line(img,detection[2],detection[3],(0,0,255),2)
        cv2.line(img,detection[3],detection[0],(0,0,255),2)


cv2.imshow("Detection results",img)

t2 = time.time()
#(rects, weights) = hog.detectMultiScale(image,scale=1.05)
(rects, weights) = hog.detect(image,padding=(8,8))
t3 = time.time()
if len(rects):
    #for (x,y,w,h) in rects:
    #    cv2.rectangle(image,(x,y), (x+w, y+h), (0,255,0), 2)
    for (x,y) in rects:
        cv2.rectangle(image,(x,y), (x+64, y+128), (0,255,0), 2)
        
print t3-t2
cv2.imshow("Original HOG no scale", image)
cv2.waitKey(0)

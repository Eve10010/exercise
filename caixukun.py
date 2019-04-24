# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 08:36:40 2019

@author: asus
"""

##蔡徐坤打篮球的代码视频

import time
import cv2
import os


a,b = os.get_terminal_size()
SIZE = (int(0.9*a),b)
TIME_SLEEP = 0.04


print("terminal size is:{} \n time sleep is:{}".format(SIZE,TIME_SLEEP))
time.sleep(2)
#read picture
if __name__ =="__main__":
     img_path = "timg.jpg"
     img = cv2.imread(img_path,0)
     img = img.reshape((1,-1)).squeeze()
     
def get_str(img0):
    img1 = img0.reshape((1,-1)).squeeze()
    img = img1/256*10
    string1 = []
    for i in range(len(img)):
        string1.append(str(int(img[i])))
    return string1

#read video
    

def readvideo(filepath):
    cap = cv2.VideoCapture(filepath)
    while(True):
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame,SIZE)
        data = get_str(frame)
        i = 0
        for letter in data:
            i += 1
            if i % SIZE[0] == 0:
                print(letter,sep='')

            else:
                print(letter,sep='',end='')

        time.sleep(TIME_SLEEP)
    return 0

readvideo('86843226-1-6.mp4')
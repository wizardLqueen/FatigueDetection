import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
workPath='/Users/apple/Desktop/FatigueDecetion/'
picPath='/Users/apple/Desktop/FatigueDecetion/FACE_BAG/MultiPIE/'
sleepPath=workPath+'TMP/Picture/Sleep/'
awakePath=workPath+'TMP/Picture/Awake/'
shapePath=workPath+'shape_predictor_68_face_landmarks.dat'
testPath = '/Users/apple/Desktop/FatigueDecetion/FACE_BAG/yalefaces__/'
trainPath = '/Users/apple/Desktop/FatigueDecetion/FACE_BAG/MultiPIE/train_image128x128/'
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(shapePath)
win = dlib.image_window()
#win.clear_overlay()

All_mark=[]
train_cnt=2019
test_cnt=47
def pre(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=cv2.GaussianBlur(img,(3,3),0,0)
    return img

def mark(img):
    global All_mark

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print '###'
    win.set_image(img1)
    rects = detector(img1, 1,)
    if len(rects)!=1:
        return False
    for i, j in enumerate(rects):
        ech=[]
        rects[i]=dlib.rectangle(0,0,img1.shape[1],img1.shape[0])
        #print rects[i]
        cv2.rectangle(img, (rects[i].left(), rects[i].top()), (rects[i].right(), rects[i].bottom()), color=(0, 255, 0))
        shape = landmark_predictor(img1,rects[i])
        win.add_overlay(shape)
        for i in range(68):
            #if i >= 48 and i <= 59:
            #    continue
            ech.append(shape.part(i).x)
            ech.append(shape.part(i).y)
            cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0))
            cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                        color=(255, 255, 255))
        All_mark.append(ech)
    win.add_overlay(rects)
    return True

show = [
4, 9, 15, 23, 30, 31, 34, 38, 43, 44,]
def change_test():
    for i in range(test_cnt):
        #print 'now '+ str(i)
        img=cv2.imread(testPath+str(i+1)+'.jpg')
        img = cv2.resize( img , (128,128) )
        img = pre(img)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
       # print '$$$'
        if mark(img) == False:
            print str(i+1)+',',
        """
        cv2.imshow('GG',img)
        key=cv2.waitKey(0)&0xff
        win.clear_overlay()
        if key == ord('q'):
            break
        """
def change_train():
    for i in range(train_cnt):
        #print 'now '+ str(i)
        img=cv2.imread(trainPath+str(i+1)+'.jpg')
        img = cv2.resize( img , (128,128) )
        img = pre(img)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
       # print '$$$'
        if mark(img) == False:
            print str(i+1)+',',

        cv2.imshow('GG',img)
        key=cv2.waitKey(0)&0xff
        win.clear_overlay()
        if key == ord('q'):
            break

def out_test():
    with open(testPath+'test_ch_mark.txt','w') as file:
        for i in range(All_mark.__len__()):
            for j in range(All_mark[i].__len__()):
                file.write(str(All_mark[i][j])+' ')
            file.write('\n')


def out_train():
    with open(trainPath+'ch_mark.txt','w') as file:
        for i in range(All_mark.__len__()):
            for j in range(All_mark[i].__len__()):
                file.write(str(All_mark[i][j])+' ')
            file.write('\n')

video_capture = cv2.VideoCapture(0)
while True:
    ret,img1=video_capture.read()

""""
change_train()
out_train()
All_mark=[]
change_test()
out_test()
"""



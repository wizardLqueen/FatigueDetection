import cv2
import pandas as pd
import time
from sklearn.pipeline import *
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import *
from sklearn.svm import *
from sklearn.externals import joblib
from sklearn.externals.joblib import *
import skflow
from features import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import dlib
import numpy as np
import img_process
from progressbar import *
import features
import random
from PATH import *
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(SHAPE_PATH)
MODEL_NAME = ''
CELL_SIZE = (32,32)
faceCascade=cv2.CascadeClassifier(CASCADE_PATH)
video_capture=cv2.VideoCapture(0)
svm = sklearn.externals.joblib.load('split_lbp_svm.m')

def pre(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img,(5,5),0,0)
    return img
def detectFace(img):
    img = pre(img)
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(100, 100),
    )
    return faces
def FatigueDetect(img,pos):
   # print 'Detect...'
    print pos
    faces = detector(img,0)
    #face_shape = dlib.rectangle(left = pos[0],top = pos[1],bottom = pos[0]+pos[2],right = pos[0]+pos[3])
    shape = landmark_predictor(img,faces[0])
    #img,shape= img_process.getRotImg(img,shape)
    left_eye,t= img_process.getLeftEye(img,shape)
    right_eye,t = img_process.getRightEye(img,shape)
    mouth,t = img_process.getMouth(img,shape)
    left_eye = cv2.resize(left_eye,(64,32))
    right_eye = cv2.resize(right_eye,(64,32))
    mouth = cv2.resize(mouth,(64,64))
    cv2.imshow('LeftEye',left_eye)
    cv2.imshow('RightEye',right_eye)
    cv2.imshow('Mouth',mouth)
    for i in range(68):
        cv2.circle(img,(shape.part(i).x,shape.part(i).y),2,(255,255,255),2)

    #print 'getLbp...'
    left_lbp = img_process.getLBPFeatures(left_eye,CELL_SIZE)
    right_lbp = img_process.getLBPFeatures(right_eye,CELL_SIZE)
    mouth_lbp = img_process.getLBPFeatures(mouth,CELL_SIZE)
    feature = np.concatenate((left_lbp,right_lbp,mouth_lbp))
    #feature = feature.reshape(-1,1)
    #print 'get...End'
    #print 'Detect...End'
    return svm.predict(feature.reshape(1,-1))
if __name__  == '__main__':
    print type(svm)
    flag = False
    face_x = 0
    face_y = 0
    face_w = 0
    face_h = 0
    while True:
        ret,frame=video_capture.read()
       # print frame.shape
        frame = cv2.resize(frame,(430,240))
        if ret!=True:
            print "Open Failed"
            continue
        ROI=frame[:,100:400]
        try:
            if flag == False:
                face = ROI
                for (x,y,w,h) in detectFace(ROI):
                    face_x = x
                    face_y = y
                    face_h = h
                    face_w = w
                    flag = True
                print flag
            if flag == True:
                face = ROI[face_y-20:face_y+face_h+20,face_x-20:face_x+face_w+20]
            else:
                face = ROI
            face = cv2.resize(face, (320, 320))
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            face = cv2.equalizeHist(face)
            text = 'Awake'
            if FatigueDetect(face,(face_x,face_y,face_w,face_h))[0] == 0:
                text = 'Fatigue'
            cv2.putText(face,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        except:
            print 
        cv2.imshow("Frame", ROI)
        cv2.imshow('Face',face)
        key=cv2.waitKey(0)&0xFF
        if key==ord('q'):
            print 'Q'
            break
  

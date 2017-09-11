import cv2
import pandas as pd
from sklearn.pipeline import *
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import *
from sklearn.svm import *
from sklearn.externals import joblib
import skflow
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn import cross_validation
from sklearn import metrics

import dlib
import numpy as np
workPath = '/Users/apple/Desktop/FatigueDecetion/'
trainPath = '/Users/apple/Desktop/FatigueDecetion/FACE_BAG/MultiPIE/train_image128x128/'
testPath = '/Users/apple/Desktop/FatigueDecetion/FACE_BAG/yalefaces__/'
sleepPath = workPath + 'TMP/Picture/Sleep/'
awakePath = workPath + 'TMP/Picture/Awake/'
shapePath = workPath + 'shape_predictor_68_face_landmarks.dat'
modelPath = '/Users/apple/Desktop/FatigueDecetion/Model/SVM/CH_SVM/'
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(shapePath)
w = 128
h = 128
c = 1
threshold_bs = 41
threshold_c = 17
clf_name = ''
train_cnt = test_cnt = 0
x_train = np.zeros((1,1))
y_train = []
x_test = np.zeros((1,1))
y_test = []
nec_pre = True
components=0
def preData():
    global x_test,x_train,train_cnt,test_cnt,x_test,y_test,y_train
    test_cnt=47
    x_train = np.loadtxt(trainPath+'ch_mark.txt')
    #print x_train
    train_cnt = x_train.__len__()
    y_train = [1 for i in range(train_cnt)]
    with open ( trainPath + 'FatigueNum_.txt' , 'r' ) as file:
        fatigue_num = file.read().split()
        for i in range(fatigue_num.__len__()):
            y_train[int(fatigue_num[i])-1] = 0
    #print y_train
    x_test=np.loadtxt(testPath+'test_ch_mark.txt')
    #print x_test
    y_test = [1 for i in range(test_cnt)]
    for i in range(23,test_cnt,1):
        y_test[i] = 0
    print x_train.shape ,y_train.__len__()
    print x_test.shape , y_test.__len__()
    #print y_test

video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(shapePath)
clf = joblib.load('/Users/apple/Desktop/FatigueDecetion/Model/SVM/CH_SVM/rbf_PCA_60_C_1_gamma_0.01_no_f_svm.m')


def pre(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0, 0)
    return img
def mark(img,ori_img):
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
   # cv2.imshow('GG',img)
   # cv2.waitKey(0)
    faces = detector(img,1)
   # print len(faces)
    vec = []
    if len(faces) !=1:
        return None,False
    for k,d in enumerate(faces):
        shape = landmark_predictor(img,d)
        cv2.rectangle(ori_img,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
        img_tmp = img[d.top():d.bottom(),d.left():d.right()]
        M = cv2.getRotationMatrix2D(center=(img_tmp.shape[0]/2,img_tmp.shape[1]/2),angle=0,scale=1.1)
        img_tmp = cv2.warpAffine(img_tmp,M,(img_tmp.shape[0],img_tmp.shape[1]))
        img_tmp = cv2.resize(img_tmp,(128,128))
        d1 = dlib.rectangle(0,0,128,128)
        shape1 = landmark_predictor(img_tmp,d1)
        for i in range(68):
            cv2.circle(ori_img,(shape.part(i).x,shape.part(i).y),1,(0,255,0),-1,8)
            cv2.circle(img_tmp, (shape1.part(i).x, shape1.part(i).y), 1, (0, 255, 0), -1, 8)
            if i > 16:
                vec.append( shape1.part(i).x )
                vec.append( shape1.part(i).y )
        cv2.imshow('GG', img_tmp)
    return vec,True
preData()
print x_train.shape, y_train.__len__()
print x_test.shape, y_test.__len__()
clf = Pipeline([('pca',PCA(n_components=60)),('ss',StandardScaler()),('svc',SVC(kernel='rbf',C=1,gamma=0.01))])
clf.fit(x_train,y_train)
while True:
    rect,frame=video_capture.read()
    frame = cv2.resize(frame,(320,180))
    img = pre(frame)
    vec,flag=mark(img,frame)
    if flag:
        print vec
        #print np.mat(vec).shape
        predict = clf.predict(np.mat(vec))
        txt=' '
        if predict[0]==0:
            txt='Fatigue'
        else:
            txt='Awake'
        print predict
        cv2.putText(frame, txt, (10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255))
    cv2.imshow('Frame', frame)
    key=cv2.waitKey(0)&0xff
    if key==ord('q'):
        break


import cv2
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import *
from sklearn.svm import *
from sklearn.preprocessing import *
import skflow
import pandas as pd
import numpy as np
workPath='/Users/apple/Desktop/FatigueDecetion/'
sleepPath=workPath+'TMP/Picture/Sleep/'
awakePath=workPath+'TMP/Picture/Awake/'
data=[]
X_train=[]
Y_train=[]
X_test=[]
Y_test=[]
n=0
m=0
def pre(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img,(5,5),0,0)
    return img
for i in range(1,43,1):
    for u in range(2):
        img=None
        if u==0:
            if i<41:
                Y_train.append(1)
            else:
                Y_test.append(1)
            img=cv2.imread(sleepPath+str(i)+'.jpeg')
        if(u==1) :
            if i < 41:
                Y_train.append(-1)
            else:
                Y_test.append(-1)
            img=cv2.imread(awakePath+str(i)+'.jpeg')
        print img.shape,i
        img=pre(img)
        n,m=img.shape[0:2]
        X=[]
        for j in range(n):
            for k in range(m):
                X.append(img[j][k])
        if i <= 40:
            data.append(X)
        else:
            X_test.append(X)
data=np.mat(data)
X_train=data
Y_test.append(1)
img=cv2.imread('/Users/apple/Desktop/1.jpeg',0)
img=cv2.resize(img,(320,300))
X=[]
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        X.append(img[i][j])
X_test.append(X)
print data.shape
mean_x=data.mean(axis=0)
plt.gray()
plt.subplot(2,4,1)
plt.imshow(mean_x.reshape(n,m))
test=X_test
for i in range(10,70,10):
    pca=PCA(n_components=i,whiten=True)
    X_train_reduce=pca.fit_transform(data)
    get=pca.components_
    ss=StandardScaler()
    X_train_reduce=ss.fit_transform(X_train_reduce)
    Y_train=ss.fit_transform(Y_train)
    X_test=test
    print Y_train
    svc=SVC()
    print 'Training.......'
    svc.fit(X_train_reduce,Y_train)
    print 'Train End'
    X_test=np.mat(X_test)
    print X_test.shape
    X_test=pca.transform(X_test)
    X_test=ss.fit_transform(X_test)
    print X_test.shape
    print 'Predicting......'
    y_predict=svc.predict(X_test)
    print 'Predict End'
    print str(i)+":"
    print y_predict
    print Y_test
for i in range(7):
    plt.subplot(2,4,i+2)
    plt.imshow(get[i].reshape(n,m))
plt.show()



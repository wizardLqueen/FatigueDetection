import cv2
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import *
from sklearn.svm import *
from sklearn.externals import joblib
import skflow
from sklearn.metrics import *
from sklearn.preprocessing import *
import numpy as np
workPath='/Users/apple/Desktop/FatigueDecetion/'
picPath='/Users/apple/Desktop/FatigueDecetion/FACE_BAG/MultiPIE/'
sleepPath=workPath+'TMP/Picture/Sleep/'
awakePath=workPath+'TMP/Picture/Awake/'
print 'PROC_DATA......'
fatigueNum=[]

for i in range(83):
    print i
    id=1
    if i < 41:
        path=sleepPath
        id=i
    else:
        id=i-41
        path=awakePath
    img=cv2.imread(path+str(id+1)+'.jpeg',0)
    M = cv2.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2),0,1.3)
    img1 = cv2.warpAffine(img,M,img.shape)
    cv2.imwrite(path+str(id+1)+'.jpeg',img1)

"""""
for i in range(40):
    img=cv2.imread(sleepPath+str(i+1)+'.jpeg',0)
    show=img
    img=cv2.resize(img,(128,128))
    show=img
    for i in range(1,11):
        M=cv2.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2),0,1+i*1.0/10.0)
        show=np.column_stack((show,cv2.warpAffine(img,M,img.shape)))
    cv2.imshow('GG',show)
    key=cv2.waitKey(0)&0xFF
    if key==ord('q'):
        break
cv2.destroyAllWindows()
"""
"""
for i in range(1515):
    print i
    img=cv2.imread(picPath+'train_image/'+str(i+1)+'.jpg',0)
    img=cv2.resize(img,(128,128))
    cv2.imwrite(picPath+'train_image128x128/'+str(i+1)+'.jpg',img)
with open(picPath+'FatigueNum.txt','r') as file:
    fatigueNum=file.read().split()
Cnt=1516
out_str=" "
for i in range(fatigueNum.__len__()):
    img=cv2.imread(picPath+'train_image128x128/'+str(fatigueNum[i])+'.jpg',0)
    M = cv2.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2),5,1.1)
    img1=cv2.warpAffine(img,M,img.shape)
    M=cv2.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2),-5,1.1)
    img2=cv2.warpAffine(img,M,img.shape)
    out_str=out_str+str(Cnt)+' '
    cv2.imwrite(picPath+'train_image128x128/'+str(Cnt)+'.jpg',img1)
    Cnt+=1
    out_str=out_str+str(Cnt)+' '
    cv2.imwrite(picPath+'train_image128x128/'+str(Cnt)+'.jpg',img2)
    Cnt+=1
print Cnt
with open(picPath+'FatigueNum.txt','a+') as file:
    file.write(out_str)
"""

"""
def pre(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img,(3,3),0,0)
    return img

with open(workPath + 'image_mat.txt', 'w') as file:
    for i in range(1, 2074):
        print i
        img = cv2.imread(picPath + 'train_image128x128/' + str(i) + '.jpg')
        img=pre(img)
        n, m = img.shape
        for j in range(n):
            for k in range(m):
                file.write(str(img[j][k]) + ' ')
        file.write('\n')
    print "PROC_END______"

"""
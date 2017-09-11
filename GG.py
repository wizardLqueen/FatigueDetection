import cv2
import pandas as pd
import sklearn
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import *
from sklearn.svm import *
from sklearn.externals import joblib
import skflow
from sklearn.metrics import *
from sklearn.preprocessing import *
import random
import shutil
import numpy as np
workPath='/Users/apple/Desktop/FatigueDecetion/'
picPath='/Users/apple/Desktop/FatigueDecetion/FACE_BAG/MultiPIE/'
sleepPath=workPath+'TMP/Picture/Sleep/'
awakePath=workPath+'TMP/Picture/Awake/'
trainPath = '/Users/apple/Desktop/FatigueDecetion/FACE_BAG/MultiPIE/train_image128x128/'
unfaces=[297, 749, 849, 850, 853, 859, 948, 958, 1524, 1525, 1531, 1542, 1543, 1544, 1570, 1571, 1572, 1573, 1580, 1590, 1591, 1592, 1634, 1642, 1665, 1675, 1706, 1710, 1711, 1730, 1731, 1736, 1737, 1766, 1767, 1776, 1777, 1780, 1781, 1786, 1804, 1805, 1826, 1827, 1838, 1946, 1984, 2021, 2026, 2046, 2060, 2062, 2064, 2065,3000
]
Cnt=1
emo=['centerlight',
'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight',
'sad', 'sleepy', 'surprised',  'wink']
fatigue_num=[]
out_fat_num=[]
with open(picPath+'FatigueNum.txt','r') as file:
    fatigue_num=file.read().split()
for i in range(fatigue_num.__len__()):
    num=int(fatigue_num[i])
    if unfaces.count(num)>1:
        continue
    index=0
    for j in range(unfaces.__len__()):
        if unfaces[j]>num:
            index=j
            break
    out_fat_num.append(num-index)
with open(trainPath+'FatigueNum_.txt','w') as file:
    for i in range(out_fat_num.__len__()):
        file.write(str(out_fat_num[i])+' ')

Cnt=1
for i in range(1,2074):
    if unfaces.count(i)!=0:
        continue
    shutil.copyfile(picPath+'train_image128x128_/'+str(i)+'.jpg',picPath+'train_image128x128/'+str(Cnt)+'.jpg')
    Cnt+=1

""""
while Cnt<=24:
    index=random.randint(1,165)
    if (index-9)%11==0:
        continue
    else:
        shutil.copyfile(picPath+'yalefaces_/'+str(index)+'.jpg',picPath+'yalefaces__/'+str(Cnt)+'.jpg')
        Cnt+=1
for i in range(1,177,1):
    if (i-9)%11==0 or i>165:
        shutil.copyfile(picPath + 'yalefaces_/' + str(i) + '.jpg', picPath + 'yalefaces__/' + str(Cnt) + '.jpg')
        Cnt+=1"""
""""
for i in range(3,16,1):
    id=str(i)
    if i < 10:
        id='0'+str(i)
    for j in range(emo.__len__()):
        name='subject'+id+'.'+emo[j]
        os.rename(picPath+'yalefaces/'+name,str(Cnt)+'.jpg')
        Cnt+=1
"""

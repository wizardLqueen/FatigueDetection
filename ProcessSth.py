import cv2
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import *
from sklearn.svm import *
from sklearn.externals import joblib
import skflow
from sklearn.metrics import *
from tensorflow.contrib.layers import l2_regularizer
from sklearn.preprocessing import *
import tensorflow as tf
from wyrm import *
import numpy as np
modelPath = '/Users/apple/Desktop/FatigueDetection/Model/SVM/CH_SVM/'
workPath='/Users/apple/Desktop/FatigueDetection/'
picPath='/Users/apple/Desktop/FatigueDetection/FACE_BAG/MultiPIE/train_image128x128/'
sleepPath=workPath+'TMP/Picture/Sleep/'
awakePath=workPath+'TMP/Picture/Awake/'
img = cv2.imread(picPath+'1.jpg',0)
img = cv2.equalizeHist(img)
img_g = np.zeros((20,128,128))
img1 = cv2.GaussianBlur(img,(7,7),0,0)
img2 = cv2.GaussianBlur(img,(9,9),0,0)
img3 = img2 - img1
plt.gray()
plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(img1)
plt.subplot(2,2,3)
plt.imshow(img2)
plt.subplot(2,2,4)
plt.imshow(img3)
plt.show()
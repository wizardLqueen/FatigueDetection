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
from sklearn import cross_validation
from sklearn import metrics
import numpy as np
workPath='/Users/apple/Desktop/FatigueDecetion/'
picPath='/Users/apple/Desktop/FatigueDecetion/FACE_BAG/MultiPIE/'
sleepPath=workPath+'TMP/Picture/Sleep/'
awakePath=workPath+'TMP/Picture/Awake/'
w=128
h=128
c=1
threshold_bs=41
threshold_c=17
components=1500
all_cnt=2073
test_cnt=50
nec_pre=True
def pre(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=cv2.GaussianBlur(img,(3,3),0,0)
    return img
x_train=np.zeros((1,1))
y_train=[]
x_test=np.zeros((1,1))
y_test=[]
ss=StandardScaler()
def preData():
    global x_train,y_train,x_test,y_test
    x_train=np.loadtxt(workPath+'image_mat.txt')
    y_train=[1 for i in range(0,all_cnt)]
    with open(picPath + 'FatigueNum.txt','r') as file:
        num=file.read().split()
        for i in range(num.__len__()):
            y_train[int(num[i])-1]=0
    y_test = [1 for i in range(test_cnt)]
    x_test = np.zeros((test_cnt, w * h))
    for i in range(0,test_cnt):
        if i>23:
            y_test[i]=0
        img=cv2.imread('/Users/apple/Desktop/FatigueDecetion/FACE_BAG/yalefaces__/'+str(i+1)+'.jpg')
        img=cv2.resize(img,(w,h))
        img1=img
        img=pre(img)
        n,m=img.shape
        for j in range(n):
            for k in range(m):
                x_test[i,j*m+k]=img[j][k]
def pca_show(pca):
    v_ = pca.components_
    plt.gray()
    plt.subplot(2, 4, 1)
    plt.imshow(x_train.mean(axis=0).reshape(w, h))
    for i in range(7):
        plt.subplot(2, 4, i + 2)
        plt.imshow(v_[i].reshape(w, h))
    plt.show()
def test():
    global components, ss, c,nec_pre
    if nec_pre:
        preData()
        nec_pre=False
    print 'PROC_DATA_END>>>>>>'
    pca = PCA(n_components=components, whiten=True)
    print 'PCA>>>>>>'
    print 'X_TRAIN_SHAPE: ', x_train.shape
    x_train_reduce = pca.fit_transform(x_train)
    pca_show(pca)
    linersvc = joblib.load(workPath+'Model/SVM/LinerSVM/pca_'+str(components)+'_C_'+str(c)+'_linersvm.m')
    x_train_reduce=ss.fit_transform(x_train_reduce)
    x_test_reduce = pca.transform(x_test)
    x_test_reduce = ss.transform(x_test_reduce)
    print 'PREDICT>>>>>>'
    y_predict=linersvc.predict(x_test_reduce)
    print 'PREDICT_END>>>>>>'
    scores = cross_validation.cross_val_score(linersvc, x_train_reduce, y_train, cv=10)
    test_ac=linersvc.score(x_test_reduce,y_test)
    cross_ac=scores.mean()
    cross_er=scores.std()*2
    out_str='The Accuracy of SVC_PCA'+str(components)+'_C'+str(c)+' is '+str(test_ac)+"\n"
    out_str=out_str+str(classification_report(y_test,y_predict,target_names=['0','1']))+"\n"
    with open(workPath+'Model/SVM/LinerSVM/LinerSVM_PCA_Report.txt','a+') as file:
        file.write(out_str)
    print 'The Accuracy of LinerSVC_PCA'+str(components)+'_C_'+str(c)+' is ', test_ac
    #target_name=np.mat([0,1]).T
    print classification_report(y_test,y_predict,target_names=['0','1'])
    with open(workPath+'Model/SVM/LinerSVM/Cross_Ac.txt','a+') as file:
        file.write(str(test_ac)+' '+str(cross_ac)+' '+str(cross_er)+"\n")
    """plt.gray()
    Cnt = 1
    for i in range(y_predict.__len__()):
        if y_predict[i] != y_test[i]:
            if i < 41:
                plt.subplot(5, 6, Cnt)
                plt.imshow(cv2.imread(sleepPath + str(i + 1) + '.jpeg', 0))
            else:
                plt.subplot(5, 6, Cnt)
                plt.imshow(cv2.imread(sleepPath + str(i - 41 + 1) + '.jpeg', 0))
            Cnt += 1
    plt.show()"""
if __name__ == '__main__':
    global components,ss,c,nec_pre
    components=1700
    c=0.7
   # test()
    for i in range(100,3100,100):
        components=i
        test()
        """
        print 'PROC_DATA>>>>>>'
        if nec_pre:
            preData()
            nec_pre=False
        print 'PROC_DATA_END>>>>>>'
        pca=PCA(n_components=components,whiten=True)
        print 'PCA>>>>>>'
        print 'X_TRAIN_SHAPE: ' , x_train.shape
        x_train_reduce=pca.fit_transform(x_train)
        print 'X_TRAIN_SHAPE: ',x_train.shape
        print 'X_TRAIN_REDUCE_SHAPE: ',x_train_reduce.shape
        v_=pca.components_
        plt.gray()
        plt.subplot(2,4,1)
        plt.imshow(x_train.mean(axis=0).reshape(w,h))
        for i in range(7):
            plt.subplot(2,4,i+2)
            plt.imshow(v_[i].reshape(w,h))
        #plt.show()
        x_train_reduce=ss.fit_transform(x_train_reduce)
       # scores=cross_validation.cross_val_score(LinearSVC(C=c),x_train_reduce,y_train,cv=10)
       # print("LinearSVM_PCA_"+str(components)+'_C_'+str(c)+"_Accuracy: %0.10f (+/- %0.10f)" % (scores.mean(), scores.std() * 2))
       # print scores
        ss=StandardScaler()
        linersvc=LinearSVC(C=c)
        print 'TRAINING>>>>'
        print y_train
        x_train_reduce=ss.fit_transform(x_train_reduce)
        print y_train
        linersvc.fit(X=x_train_reduce,y=y_train)
        print 'SAVEING>>>>>>'
        joblib.dump(linersvc,workPath+'Model/SVM/LinerSVM/pca_'+str(components)+'_C_'+str(c)+'_linersvm.m')
        print 'END>>>>>> '
        test()
        """

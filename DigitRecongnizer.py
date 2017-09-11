import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import skflow
from sklearn import *
from sklearn.svm import *
from sklearn.preprocessing import StandardScaler
def Out(predict,Name):
    out=pd.DataFrame({'ImageId':range(1,predict.size+1),'Label':predict})
    out.to_csv(Path+Name+'.csv')
Path='/Users/apple/Desktop/Kaggle/DigitRecognizer/'

def LSVC(x_train,y_train,test):
    lsvc=LinearSVC()
    lsvc.fit(x_train,y_train)
    y_predict=lsvc.predict(test)
    Out(y_predict,'LinearSVC')
def KNC(x_train,y_train,test):
    from sklearn.neighbors import KNeighborsClassifier
    knc=KNeighborsClassifier()
    knc.fit(x_train,y_train)
    y_predict=knc.predict(test)
    Out(y_predict,'KNC')
def SVCC(x_train,y_train,test):
    svc=SVC()
    svc.fit(x_train,y_train)
    y_predict=svc.predict(test)
    Out(y_predict,'SVC')
if __name__ == '__main__':
    train = pd.read_csv(Path + 'train.csv')
    print train.shape
    test = pd.read_csv(Path + 'test.csv')
    print test.shape
    y_train = train['label']
    print y_train
    x_train = train.drop('label', 1)
    print x_train
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    test = ss.fit_transform(test)
    Cnt=[i for i in range(0,10)]
    train=train.cumsum()
    train.plot()
    #for i in range(y_train.size) :
    #    Cnt[int(y_train[i])]+=1
    #for i in range(10):
    #    print str(i)+" "+str(Cnt[i])



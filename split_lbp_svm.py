import cv2
import pandas as pd
import time
from sklearn.pipeline import *
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import *
from sklearn.svm import *
from sklearn.externals import joblib
import skflow
from features import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn import cross_validation
from sklearn.lda import *
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
face = dlib.rectangle(left = 0,top = 0,bottom = 320,right = 320)
MODEL_NAME = 'g_0.1_n_15_c_10.m'
CELL_SIZE = (32,32)
class block:
    id = 0
    label = 0
    def __init__(self,id,label):
        self.id = id
        self.label = label

X_train = []
Y_train = []
All_block = []
def preData():
    pbr = ProgressBar(maxval = 660)
    global All_block,X_train,Y_train
    now = time.time()
    print 'Data process...'
    for i in range(440):
        All_block.append(block(i,1))
    for i in range(220):  
        All_block.append(block(i+440,0))
    random.shuffle(All_block)
    pbr.start()
    for i in range(660):
        pbr.update(i)
        id = All_block[i].id
        label = All_block[i].label
        Y_train.append(label)
        path = AWAKE_PATH
        if label == 0:
            id -= 440
            path  = SLEEP_PATH
        left_eye = cv2.imread(path+'LeftEyes/'+str(id)+'.jpeg',0)
        right_eye = cv2.imread(path + 'RightEyes/' + str(id) + '.jpeg',0)
        mouth  = cv2.imread(path + 'Mouth/' + str(id) + '.jpeg',0)
        left_lbp = img_process.getLBPFeatures(left_eye,CELL_SIZE)
        right_lbp = img_process.getLBPFeatures(right_eye,CELL_SIZE)
        mouth_lbp = img_process.getLBPFeatures(mouth,CELL_SIZE)
        feature = np.concatenate((left_lbp,right_lbp,mouth_lbp))
        X_train.append(feature)
    pbr.finish()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train,dtype = 'int')
    np.savez('Train_and_Label.npz',X_train = X_train,Y_train = Y_train)
    print X_train.shape
    print 'End\nCost Time:' + str(time.time() - now)


def train():
    global X_train,Y_train,MODEL_NAME
    #print X_train,Y_train
    #pca = PCA(n_components=100)
    #tmp = pca.fit_transform(X_train)
    clf = Pipeline([
                        ('ss',StandardScaler()),
                        ('pca',PCA(whiten = True)),
                        ('lda',LDA(n_components = 1)),
                        ('svc',SVC())
                    ])
    par = [{
                'svc__kernel':['rbf','linear'],
                'svc__gamma':np.logspace(-3,0,4),
                'svc__C':np.logspace(0,1,3),
                'pca__n_components':[int(i) for i in range(5,51,3)]
          }]
    print par
    gs = GridSearchCV(clf,param_grid = par,cv = 3,refit = True,verbose = 1,n_jobs = -1 )
    gs.fit(X_train,Y_train)
    print("Best parameters set found on development set:\n")
    print gs.best_params_
    print ''
    print 'Grid scores on development set:+\n'
    best_score = 0
    best_std = 0
    for parms,mean_score,scores in gs.grid_scores_:
        if parms == gs.best_params_:
            best_score = mean_score
            best_std = scores.std()*2
        print "%0.5f (+/-%0.5f) for %r" %(mean_score,scores.std()*2,parms)
    print ''
    print 'Detailed Report+\n'
    write_par = ("%0.5f (+/-%0.5f) for %r\n" %(best_score,best_std,gs.best_params_))
    print "%0.5f (+/-%0.5f) for %r" %(best_score,best_std,gs.best_params_)
    print 'The model is trained on the full development set.'
    print 'The score are computed on the full evaluation set\n'
    MODEL_NAME = 'split_lbp_svm.m'
    with open('Best_Par.txt','w+') as file:
        file.write(write_par)
    joblib.dump(gs,MODEL_NAME)

X_test = []
Y_test = []
def preTest():
    global X_test,Y_test
    for i in range(1,31):
        id = i
        label = 1
        path = TEST_PATH + 'Awake/'
        if i > 23:
            path = TEST_PATH + 'Sleep/'
            id -= 23
            label = 0
        img = cv2.imread(path+str(id)+'.jpeg',0)
        img = cv2.resize(img,(320,320))
        shape = landmark_predictor(img,face)
        img,shape= img_process.getRotImg(img,shape)
        left_eye,t= img_process.getLeftEye(img,shape)
        right_eye,t = img_process.getRightEye(img,shape)
        mouth,t = img_process.getMouth(img,shape)
        left_eye = cv2.resize(left_eye,(64,32))
        right_eye = cv2.resize(right_eye,(64,32))
        mouth = cv2.resize(mouth,(64,64))
        """
        cv2.imshow('img',img)
        cv2.imshow('GG1',left_eye)
        cv2.imshow('GG2',right_eye)
        cv2.imshow('GG3',mouth)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        """
        left_lbp = img_process.getLBPFeatures(left_eye,CELL_SIZE)
        right_lbp = img_process.getLBPFeatures(right_eye,CELL_SIZE)
        mouth_lbp = img_process.getLBPFeatures(mouth,CELL_SIZE)
        feature = np.concatenate((left_lbp,right_lbp,mouth_lbp))
        X_test.append(feature)
        Y_test.append(label)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

def test():
    global X_test,Y_test
    clf = joblib.load(MODEL_NAME)
    Y_predict = clf.predict(X_test)
    print 'The Accuracy of the Model is ' + str(clf.score(X_test, Y_test))
    print classification_report(Y_test, Y_predict)
    print ''

def loadData():
    global X_train,Y_train
    data = np.load('Train_and_Label.npz')
    X_train = data['X_train']
    Y_train = data['Y_train']
    #print X_train.shape,Y_train
def see():
    open_eye = cv2.imread(AWAKE_PATH+'LeftEyes/'+str(400)+'.jpeg',0)
    close_eye = cv2.imread(SLEEP_PATH+'LeftEyes/'+str(180)+'.jpeg',0)
    close_mouth = cv2.imread(AWAKE_PATH+'Mouth/'+str(0)+'.jpeg',0)
    open_mouth = cv2.imread(SLEEP_PATH+'Mouth/'+str(19)+'.jpeg',0)
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.gray()
    plt.imshow(LBP(open_eye))
    plt.subplot(2,2,2)
    plt.imshow(LBP(close_eye))
    plt.subplot(2,2,3)
    plt.xlim(0,256*2)
    plt.ylim(0,0.20)
    plt.bar(range(256*2),img_process.getLBPFeatures(open_eye,CELL_SIZE))
    plt.gray()
    plt.grid()
    plt.subplot(2,2,4)
    plt.xlim(0,256*2)
    plt.ylim(0,0.20)
    plt.bar(range(256*2),img_process.getLBPFeatures(close_eye,CELL_SIZE))
    plt.gray()
    plt.grid()
    plt.figure(2)
    plt.subplot(2,2,1)
    plt.gray()
    plt.imshow(LBP(open_mouth))
    plt.subplot(2,2,2)
    plt.gray()
    plt.imshow(LBP(close_mouth))
    plt.subplot(2,2,3)
    plt.gray()
    plt.grid()
    plt.xlim(0,256*4)
    plt.ylim(0,0.4)
    plt.bar(range(256*4),img_process.getLBPFeatures(open_mouth,CELL_SIZE))
    plt.subplot(2,2,4)
    plt.gray()
    plt.grid()
    plt.xlim(0,256*4)
    plt.ylim(0,0.4)
    plt.bar(range(256*4),img_process.getLBPFeatures(close_mouth,CELL_SIZE))
    plt.show()
def visualize():
    global X_train,Y_train
    clf = Pipeline(
        [
            ('ss',StandardScaler()),
            ('pca',PCA(n_components=50)),
            ('lda',LDA(n_components=1))
        ]
    )
    clf.fit(X_train,Y_train)
    X_new = clf.transform(X_train)
    print X_new
    Y_new = [0 for i in range(X_new.shape[0])]
    plt.scatter(X_new[:],Y_train[:],c=Y_train,s=25,alpha=0.4,marker='o')
    plt.show()
if __name__ == '__main__':
    np.random.RandomState(1234567)
    random.seed(1234567)
    loadData()
    visualize()
    """
    train()
    preTest()
    test()
    """
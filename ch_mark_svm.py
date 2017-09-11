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

def train_test():
    ss=StandardScaler()
    gs = Pipeline([('pca', PCA(n_components=components)), ('ss', StandardScaler()),
                    ('svc', SVC())])
   # print gs.named_steps['pca']
    pca=PCA(n_components=components)
    x_train_reduce=pca.fit_transform(x_train)
    x_train_ss=ss.fit_transform(x_train_reduce)
    from sklearn.grid_search import GridSearchCV
    par=[{'kernel':['rbf'],'gamma':np.logspace(-4,1,6),'C':np.logspace(-3,3,7)},
         #{'kernel':['linear'],'C':np.logspace(-3,1,5)}
         ]
    for i in range(par.__len__()):
        clf=GridSearchCV(SVC(C=1),param_grid=par[i],cv=5,refit=True,verbose=2,n_jobs=-1)
        clf.fit(x_train_ss,y_train)
        print("Best parameters set found on development set:\n")
        print clf.best_params_
        print ''
        print 'Grid scores on development set:+\n'
        for parms,mean_score,scores in clf.grid_scores_:
            print "%0.5f (+/-%0.5f) for %r" %(mean_score,scores.std()*2,parms)
        print ''
        print 'Detailed Report+\n'
        print 'The model is trained on the full development set.'
        print 'The score are computed on the full evaluation set\n'
        x_test_reduce=pca.transform(x_test)
        x_test_ss = ss.transform(x_test_reduce)
        y_predict = clf.predict(x_test_ss)
        print 'The Accuracy of the Model is ' + str(clf.score(x_test_ss, y_test))
        print classification_report(y_test, y_predict)
        print ''

def test(kernel,C,gamma):
    print components,kernel,C,gamma
    clf=Pipeline([('pca',PCA(n_components=components)),('ss',StandardScaler()),('svc',SVC(kernel=kernel,C=C,gamma=gamma))])
    clf.fit(x_train,y_train)
    out_str=''
    y_predict = clf.predict(x_test)
    out_str+=(str(clf.score(x_test,y_test))+' ')
    print out_str
    if clf.score(x_test,y_test) > 0.93:
        score = cross_validation.cross_val_score(clf, x_train, y_train, cv=5)
        print "%0.5f (+/-%0.5f) for " % (score.mean(), score.std() * 2)
        print classification_report(y_test, y_predict)
        plt.gray()
        Cnt = 1
        for i in range(y_test.__len__()):
            if y_predict[i] != y_test[i]:
                # print str(i+1)+',',
                img = cv2.imread(testPath + str(i + 1) + '.jpg')
                plt.subplot(5, 6, Cnt)
                Cnt += 1
                plt.imshow(cv2.resize(img, (128, 128)))
        plt.show()
        joblib.dump(clf, modelPath + kernel + "_PCA_" + str(components) + "_C_" + str(C) + "_gamma_" + str(
            gamma) + '_no_f_svm.m')

        return True
    print '......'
    return False
    #plt.show()
    #with open(modelPath+kernel+'_ac.txt','a+') as file:
      #  file.write(out_str+'\n')
    joblib.dump(clf,modelPath+kernel+"_PCA_"+str(components)+"_C_"+str(C)+"_gamma_"+str(gamma)+'_no_f_svm.m')

if __name__ == '__main__':
    global components
    print 'Prepare Data>>>\n'
    preData()
    print 'Prepare End>>>\n'
    components=60
    while test('rbf',1,0.01) ==False:
        None

    #train_test()
    #for i in range(10,110,10):
    #    components=i
    #    test('rbf',1,0.02)
#-*- coding: UTF-8 -*-
import cv2
import sys
import Queue
import dlib
import tensorflow
faceCascade=cv2.CascadeClassifier('/usr/local/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_alt.xml')
eyeCascade=cv2.CascadeClassifier('/usr/local/opencv-3.2.0/data/haarcascades_cuda/haarcascade_eye_tree_eyeglasses.xml')
Path='/Users/apple/Desktop/FatigueDecetion/'
workPath='/Users/apple/Desktop/FatigueDecetion/'
picPath='/Users/apple/Desktop/FatigueDecetion/FACE_BAG/MultiPIE/'
sleepPath=workPath+'TMP/Picture/Sleep/'
awakePath=workPath+'TMP/Picture/Awake/'
shapePath=workPath+'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(shapePath)
def pre(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化图像
    img = cv2.equalizeHist(img)  # 直方图均衡化
    img = cv2.GaussianBlur(img,(3,3),0,0) #高斯滤波
    return img
def detectFace(img):
    img=pre(img)
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(170, 170),
    )
    return faces
def detectEye(img):
    img=pre(img)
    eyes = eyeCascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(10,10),
    )
    Cnt=1
    return eyes
def OpenClose(img):
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    return img
tname1='block_size'
tname2='C'
ename='Eyeroi'
EyeRoi=cv2.imread('/Users/apple/Desktop/FatigueDecetion/Pictures/1.jpeg')
Face=cv2.imread('/Users/apple/Desktop/FatigueDecetion/Pictures/1.jpeg')
def fill(img,S):
    row=img.shape[0]
    col=img.shape[1]
    #print str(row)+" "+str(col)
    Graph=[[0 for j in range(col)]for i in range(row)]
    for i in range(row):
        for j in range(col):
            if img[i,j]==0:
                Graph[i][j]=1
    q=Queue.Queue(0)
    q1=Queue.Queue(0)
    flag=[[False for j in range(col)]for i in range(row)]
    step=[[0,1],[0,-1],[-1,0],[1,0]]
    for i in range(row):
        for j in range(col):
            cnt=1
            if flag[i][j]==False and Graph[i][j]==1:
                while q.empty()==False:
                    q.get()
                while q1.empty() == False:
                    q1.get()
                q.put((i,j))
                q1.put((i, j))
                flag[i][j]=True
                while q.empty()==False:
                    (x,y)=q.get()
                    for k in range(4):
                        nowx=x+step[k][0]
                        nowy=y+step[k][1]
                        if nowx<row and nowy<col and nowx>=0 and nowy>=0 and Graph[nowx][nowy]==1 and flag[nowx][nowy]==False:
                            q.put((nowx,nowy))
                            flag[nowx][nowy]=True
                            q1.put((nowx,nowy))
            if q1.empty()==False:
                if int(q1.qsize())<=S:
                    while q1.empty() == False:
                        (x,y)=q1.get()
                        Graph[x][y]=0
                        img[x,y]=255

    return img
def showEyeRoi(arg):
    bs=cv2.getTrackbarPos(tname1,ename)
    C=cv2.getTrackbarPos(tname2,ename)
    if (bs&1)==0:
        bs+=1
    if (C&1)==0:
        C+=1
    print 'bs'+str(bs)+'\n'
    print 'C' +str(C)+'\n'
    global EyeRoi
    cv2.imshow('Tmp',EyeRoi)
    Eyeroi=cv2.adaptiveThreshold(EyeRoi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bs,C)
   # Eyeroi=fill(Eyeroi,20)
    cv2.imshow(ename,Eyeroi)

def SolveFace(img,show):
    global EyeRoi
    cv2.imshow('Face',img)
    face=img
    Eyeroi = face[face.shape[0] / 5:face.shape[0] / 2, 0:face.shape[1]]
    EyeRoi=pre(Eyeroi)
    EyeRoi=pre(face)
    eyes=detectEye(Eyeroi)
    """if show:
        cv2.imshow('EyeRoi', Eyeroi)
    cnt=1
    Eyes=[]
    for (x, y, w, h) in eyes:
        eye = Eyeroi[y + h / 6:y + h - h / 6, x - w / 8:x + w + w / 8]
        eye=pre(eye)
        Eyes.append(eye)
        if show:
            cv2.imshow("Eye" + str(cnt), eye)
            tmp = cv2.adaptiveThreshold(eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                   63,5 )
            tmp=OpenClose(tmp)
            #tmp=fill(tmp,int((tmp.shape[0]*tmp.shape[1])/5))
            getUpperEyelid(tmp,eye)
            cv2.imshow("Eye" + str(cnt), eye)
            cv2.imshow("EyeThre"+str(cnt),tmp)
        cnt+=1
    return eyes"""
Cnt=1

def getUpperEyelid(img,oriimg):
     upper=[]
     row=img.shape[0]
     col=img.shape[1]
     for i in range(col):
         for j in range(row):
             if img[j,i]==0:
                 upper.append((i,j))
                 break
     for i in range(upper.__len__()):
         cv2.circle(oriimg,upper[i],1,(255,255,255))
if __name__ == '__main__':
    op=raw_input("Chose Way:1.Video 2.Image: 3.ProcessAll: 4.Test")
    font = cv2.FONT_HERSHEY_SIMPLEX
    file=open(Path+'Pictures/Num.txt','r')
    Cnt=int(file.read())
    file.close()
    print "Cnt: "+str(Cnt)

    if op == '1':
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            face=frame
            for (x, y, w, h) in detectFace(frame):
                cv2.rectangle(frame, (x, y), (x + w, y + w), (255, 255, 255), 2)
                cv2.putText(frame, "(" + str(x) + " " + str(y) + ") " + "W: " + str(w) + " H: " + str(h), (x, y), font,
                            0.5, (255, 255, 255), 1)
                face = frame[y:y + w, x:x + w]
            cv2.imshow('Frame', frame)
            cv2.imshow('Face',face)
            #SolveFace(face)
            key=cv2.waitKey(1)&0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                def writeImg(img,name):
                    cv2.imwrite("/Users/apple/Desktop/FatigueDecetion/Pictures/" + str(Cnt) + name +".jpeg", img)
                res256=cv2.resize(face,(256,256),interpolation=cv2.INTER_AREA)
                res128=cv2.resize(face,(128,128),interpolation=cv2.INTER_AREA)
                res64=cv2.resize(face,(64,64),interpolation=cv2.INTER_AREA)
                writeImg(res256,"_256")
                writeImg(res128,"_128")
                writeImg(res64,"_64")
                file = open('/Users/apple/Desktop/FatigueDecetion/Pictures/Num.txt', 'w')
                Cnt += 1
                file.write(str(Cnt))
                file.close()
        video_capture.release()
    elif op == '2':
        frame=cv2.imread(picPath+'train_image128x128/1.jpg')
       # frame=cv2.imread('/Users/apple/Desktop/FatigueDecetion/Pictures/10.jpeg')
        cv2.imshow('ReadImage',frame)
        SolveFace(frame,True)
        cv2.namedWindow(ename)
        cv2.createTrackbar(tname1, ename, 3, 255, showEyeRoi)
        cv2.createTrackbar(tname2, ename, 3, 255, showEyeRoi)
        key=cv2.waitKey(0)
    elif op == '3':
        Temp=True
        for i in range(1,Cnt,1):
            img=cv2.imread('/Users/apple/Desktop/FatigueDecetion/Pictures/'+str(i)+'.jpeg')
            img1=pre(img)
            img1=cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,23,7)
            img1=OpenClose(img1)
            img1=fill(img1,200)
            cv2.imwrite(Path+'Pictures/Face1/'+str(i)+'.jpeg',img1)
            eyes=SolveFace(img,False)
            cnt=0
            for (x,y,w,h) in eyes:
                #eye = Eyeroi[y + h / 6:y + h - h / 6, x - w / 8:x + w + w / 8]
                eye = img1[img1.shape[0]/5+y+h/6:img1.shape[0]/5+y+h-h/6,x-w/8:x+w+w/8]
                cv2.imwrite(Path + 'Pictures/Eyes/' + str(i) + '_'+str(cnt)+'.jpeg', eye)
                cnt=(cnt+1)%2
    elif op=='4':

        cv2.destroyAllWindows()

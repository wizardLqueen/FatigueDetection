import cv2
import dlib
import numpy as np
import features
from matplotlib import pyplot
from PATH import *

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(SHAPE_PATH)

def getRectangle(st,ed,shape):
    x_tmp = []
    y_tmp = []
    for i in range(st,ed+1):
        x_tmp.append(shape.part(i).x)
        y_tmp.append(shape.part(i).y)
    return (min(x_tmp),min(y_tmp)),(max(x_tmp),max(y_tmp))

def getAngle(st,ed):
    if ed[0] == st[0]:
        return 0
    k = (ed[1]-st[1])*1.0/(ed[0]-st[0]) 
    angle = np.arctan(k)/np.arccos(-1)*180
    angle += 180 if angle <= 0  else 0
    return angle - 90 

def drawLandmark(img,shape):
    for i in range(68):
        cv2.circle(img,(shape.part(i).x,shape.part(i).y),3,(0,255,0))

def getLeftEye(img,shape):
    left_min,left_max = getRectangle(36,41,shape)
    lefteye_top_left = (max(0,min(shape.part(17).x,left_min[0])),(shape.part(17).y+left_min[1])//2)
    lefteye_bottom_right  = (max(shape.part(21).x,left_max[0]),max(shape.part(28).y,left_max[1]))
    return getRectangleImg(img,lefteye_top_left,lefteye_bottom_right),(lefteye_top_left,lefteye_bottom_right)

def getRightEye(img,shape):
    right_min,right_max = getRectangle(42,47,shape)
    righteye_top_left = (max(0,min(right_min[0],shape.part(22).x)),(right_min[1]+shape.part(22).y)//2)
    righteye_bottom_right = (max(shape.part(26).x,right_max[0]),max(shape.part(28).y,right_max[1]))
    return getRectangleImg(img,righteye_top_left,righteye_bottom_right),(righteye_top_left,righteye_bottom_right)

def getMouth(img,shape):
    mouth_min,mouth_max = getRectangle(48,59,shape)
    mouth_top_left = (max(0,mouth_min[0]),(mouth_min[1]+shape.part(33).y)//2)
    mouth_bottom_right = (mouth_max[0],(mouth_max[1]+shape.part(8).y)//2)
    return getRectangleImg(img,mouth_top_left,mouth_bottom_right),(mouth_top_left,mouth_bottom_right)

def getRotImg(img,shape):
    rot_img = img.copy()
    rot_angle = 100
    rot_shape = shape
    face = dlib.rectangle(left = 0,top = 0,bottom = 320,right = 320)
    while(abs(rot_angle)> 0.1):
        rot_angle = getAngle([rot_shape.part(27).x,rot_shape.part(27).y],[rot_shape.part(30).x,rot_shape.part(30).y])
        rot_mat = cv2.getRotationMatrix2D(center = (img.shape[0]/2,img.shape[1]/2),angle = rot_angle,scale = 1)
        rot_img = cv2.warpAffine(rot_img,rot_mat,(img.shape[0],img.shape[1]))
        rot_shape = landmark_predictor(rot_img,face)
    return rot_img,rot_shape

def getCellImg(img,cell_size):
    
    w = cell_size[0]
    h = cell_size[1]
    for i in range(0,img.shape[0]-w+1,w):
        for j in range(0,img.shape[1]-h+1,h):
            yield img[i:i+w,j:j+h]

def getRectangleImg(img,lp,rp):
    return img[lp[1]:rp[1],lp[0]:rp[0]]

def getLBPFeatures(img,cell_size):
    lbp_img = features.LBP(img)
    w = cell_size[0]
    h = cell_size[1]
    cell_cnt = w*h
    X = []
    for cell in getCellImg(lbp_img,cell_size):
        tmp = [0 for i in range(256)]
        for i in range(cell.shape[0]):
            for j in range(cell.shape[1]):
                tmp[cell[i,j]] += 1
        for i  in range(256):
            tmp[i]/=(w*h*1.0)
        X.extend(tmp)
    #print len(X)
    return np.array(X,np.float32)   
import cv2
import dlib
import numpy as np
import img_process
import features
from matplotlib import pyplot
PIC_PATH = '/Users/apple/Desktop/FatigueDetection/FACE_BAG/My_Own_Pic/TRAIN_PIC/'
SHAPE_PATH = '/Users/apple/Downloads/shape_predictor_68_face_landmarks.dat'
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

def drawRectangle(img,shape):
    left_min,left_max = getRectangle(36,41,shape)
    right_min,right_max = getRectangle(42,47,shape)
    mouth_min,mouth_max = getRectangle(48,59,shape)
    righteye_top_left = (min(right_min[0],shape.part(22).x),(right_min[1]+shape.part(22).y)//2)
    righteye_bottom_right = (max(shape.part(26).x,right_max[0]),max(shape.part(28).y,right_max[1]))
    lefteye_top_left = (min(shape.part(17).x,left_min[0]),(shape.part(17).y+left_min[1])//2)
    lefteye_bottom_right  = (max(shape.part(21).x,left_max[0]),max(shape.part(28).y,left_max[1]))
    mouth_top_left = (mouth_min[0],(mouth_min[1]+shape.part(33).y)//2)
    mouth_bottom_right = (mouth_max[0],(mouth_max[1]+shape.part(8).y)//2)
    cv2.rectangle(img,lefteye_top_left,lefteye_bottom_right,(0,255,0))
    cv2.rectangle(img,righteye_top_left,righteye_bottom_right,(0,255,0))
    cv2.rectangle(img,mouth_top_left,mouth_bottom_right,(0,255,0))

def getRotImg(img):
    face = dlib.rectangle(left = 0,top = 0,bottom = 320,right = 320)
    shape = landmark_predictor(img,face)
    rot_img = img.copy()
    rot_angle = 100
    rot_shape = shape
    while(abs(rot_angle)> 0.1):
        rot_angle = getAngle([rot_shape.part(27).x,rot_shape.part(27).y],[rot_shape.part(30).x,rot_shape.part(30).y])
        rot_mat = cv2.getRotationMatrix2D(center = (img.shape[0]/2,img.shape[1]/2),angle = rot_angle,scale = 1)
        rot_img = cv2.warpAffine(rot_img,rot_mat,(img.shape[0],img.shape[1]))
        rot_shape = landmark_predictor(rot_img,face)
        #print rot_angle
    drawLandmark(rot_img,rot_shape)
    drawRectangle(rot_img,rot_shape)
    return rot_img

for id in range(0,440):
    face = dlib.rectangle(left = 0,top = 0,bottom = 320,right = 320)
    img = cv2.imread(PIC_PATH+'Awake/'+str(id)+'.jpeg',0)
    img = cv2.resize(img,(320,320))
    shape = landmark_predictor(img,face)
    lp0,lp1 = img_process.getLeftEyeRectangle(img,shape)
    rp0,rp1 = img_process.getRightEyeRectangle(img,shape)
    mp0,mp1 = img_process.getMouthRectangle(img,shape)
    print lp0,lp1,rp0,rp1,mp0,mp1
    left_eye_img = getRectangleImg(img,lp0,lp1)
    right_eye_img = getRectangleImg(img,rp0,rp1)
    mouth_img = getRectangleImg(img,mp0,mp1)
    cv2.imshow('LEFTEYE',left_eye_img)
    cv2.imshow('RIGHTEYE',right_eye_img)
    cv2.imshow('MOUTH',mouth_img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break


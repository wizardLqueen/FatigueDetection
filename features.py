import cv2
import numpy as np
import time
def DOG(img,num):
    return cv2.GaussianBlur(img,(num+1,num+1),0,0) - cv2.GaussianBlur(img,(1,1),0,0)
def LBP(img):
    #print "(3x3) origin LBP Features"
    w = img.shape[0]
    h = img.shape[1]
    img_padding = np.zeros((w+2,h+2),np.uint8)
    img_padding[1:w+1,1:h+1] = img[0:w,0:h]
    lbp_img = np.zeros((w,h),np.uint8)
    #now = time.time()
    for x in range(1,w+1):
        for y in range(1,h+1):
            num = img_padding[x,y]
            lbp_img[x-1,y-1] |= ((1 if img_padding[x-1,y-1] >= num else 0)<<7)
            lbp_img[x-1,y-1] |= ((1 if img_padding[x,y-1] >= num else 0)<<6)
            lbp_img[x-1,y-1] |= ((1 if img_padding[x+1,y-1] >= num else 0)<<5)
            lbp_img[x-1,y-1] |= ((1 if img_padding[x+1,y] >= num else 0 )<<4)
            lbp_img[x-1,y-1] |= ((1 if img_padding[x+1,y+1] >= num else 0)<<3)
            lbp_img[x-1,y-1] |= ((1 if img_padding[x,y+1] >= num else 0)<<2)
            lbp_img[x-1,y-1] |= ((1 if img_padding[x-1,y+1] >= num else 0)<<1)
            lbp_img[x-1,y-1] |= ((1 if img_padding[x-1,y] >= num else 0)<<0) 
    #print 'Process time:',time.time() - now     
    return lbp_img
def HOG():
    return 

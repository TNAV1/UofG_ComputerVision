# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:25:22 2019

@author: Todd Nelson
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

global dst

def Display(img1):
    cv.imshow('image',img1)
    cv.waitKey(0)
    cv.destroyAllWindows()



def OffColor(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def GreyScale(img):
    global greyimg
    greyimg=img[:,:,0]
  


def Crop(img1,img2):
    global cropimg2
    cropimg2 = img2[0:img1.shape[0],0:img1.shape[1],:]  
    return cropimg2    
    

def Overlay(img1,img2,alpha):
    Crop(img1,img2)
    #overlay 2 images
    dst = cv.addWeighted(img1,alpha,cropimg2,1-alpha,0)
    cv.imshow('dst',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()



def AddLogo(img1,img3):
    # Load two images
    #img1 = cv.imread('messi5.jpg')
    #img2 = cv.imread('opencv-logo-white.png')

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img3.shape
    roi = img1[0:rows, 0:cols ]
    
    #Now create a mask of logo and create its inverse mask also
    img3gray = cv.cvtColor(img3,cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img3gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    
    # Now black-out the area of logo in ROI
    img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
    
    # Take only region of logo from logo image.
    img3_fg = cv.bitwise_and(img3,img3,mask = mask)
    
    # Put logo in ROI and modify the main image
    dst = cv.add(img1_bg,img3_fg)
    img1[0:rows, 0:cols ] = dst
    
    cv.imshow('res',img1)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
def ScaleUp(img,a):
    global res
    height, width = img.shape[:2]
    res = cv.resize(img,(a*width, a*height), interpolation = cv.INTER_CUBIC)


def Translation(img,x,y):
    global dst
    GreyScale(img)
    img=greyimg
    rows,cols = img.shape
    M = np.float32([[1,0,x],[0,1,y]])
    dst = cv.warpAffine(img,M,(cols,rows))


def Rotation(img,angle):
    global dst
    GreyScale(img)
    img=greyimg
    rows,cols = img.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),angle,1)
    dst = cv.warpAffine(img,M,(cols,rows))


def AffineTransf(img):
    global dst
    rows,cols,ch = img.shape
    
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    
    M = cv.getAffineTransform(pts1,pts2)
    
    dst = cv.warpAffine(img,M,(cols,rows))
    
  #  plt.subplot(121),plt.imshow(img),plt.title('Input')
  #  plt.subplot(122),plt.imshow(dst),plt.title('Output')
  #  plt.show()


def ProspectiveTransf(img):
    global dst
    rows,cols,ch = img.shape
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img,M,(cols,rows))
    #plt.subplot(121),plt.imshow(img),plt.title('Input')
    #plt.subplot(122),plt.imshow(dst),plt.title('Output')
    #plt.show()


def Smooth(img,x,y,factor):
    global dst
    kernel = np.ones((x,y),np.float32)/factor
    dst = cv.filter2D(img,-1,kernel)


def GausianBlurr(img,x,y,stdev):
    global blur
    blur = cv.GaussianBlur(img,(x,y),stdev)


def Gradient(img):
    GreyScale(img)
    img=greyimg
    laplacian = cv.Laplacian(img,cv.CV_64F)
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
    
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    
    plt.show()


def Canny(img,x,y):    
    GreyScale(img)
    img=greyimg
    edges = cv.Canny(img,x,y)    
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])    
    plt.show()

def BoundingBox(img):        
    #only works for pure black and white img
    ret,thresh = cv.threshold(img,127,255,0)
    im2,contours,hierarchy = cv.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv.moments(cnt)
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img,[box],0,(255,255,255),2)
    Display(img)


def FFTs():
    img = cv.imread('C:\\eng_apps\\Python\\OpenCV\\LittleRedCorvette.JPG',0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))    
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    crow=int(crow)
    ccol=int(ccol)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)    
    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])    
    plt.show()
    Display(img_back)

def TemplateMatch():    
    img_rgb = cv.imread('C:\\eng_apps\\Python\\OpenCV\\LittleRedCorvette.JPG')
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread('C:\\eng_apps\\Python\\OpenCV\\corvette.JPG',0)
    w, h = template.shape[::-1]   
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)    
    cv.imwrite('res.png',img_rgb)
    Display(img_rgb)


def HoughLines():
    img = cv.imread('C:\\eng_apps\\Python\\OpenCV\\road.JPG')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,300,apertureSize = 3)
    lines = cv.HoughLinesP(edges,1,np.pi/180,10,minLineLength=115,maxLineGap=7)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv.imwrite('houghlines5.jpg',img)
    Display(img)



img1 = cv.imread('C:\\eng_apps\\Python\\OpenCV\\LittleRedCorvette.JPG')
img2 = cv.imread('C:\\eng_apps\\Python\\OpenCV\\road.jpg')
img3 = cv.imread('C:\\eng_apps\\Python\\OpenCV\\GM.JPG') 
img4 = cv.imread('C:\\eng_apps\\Python\\OpenCV\\table.JPG',0) 
#Display(img1)
#GreyScale(img2)
#OffColor(img1)
#Crop(img1,img2)
#Display(cropimg2)
#Overlay(img1,img2,0.6)
#AddLogo(img1,img3)
#ScaleUp(img1,2)
#Translation(img1,-300,-300)
#Rotation(img1,90)
#AffineTransf(img3)
#ProspectiveTransf(img1)
#Smooth(img3,5,5,25)
#GausianBlurr(img3,5,5,0)
#Gradient(img2)
#Canny(img2,200,400)
#BoundingBox(img4)
#FFTs()
TemplateMatch()
#HoughLines()
#Display(img)

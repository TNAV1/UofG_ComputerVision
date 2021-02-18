# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:49:00 2019

@author: Todd Nelson
"""

# Run this in a command prompt or use Anaconda
# C:\Apps\Anaconda3\python.exe W:\PythonScripts\ExtrinsicCameraTransformation.py

# This is base off the Udacity Computer Vision Course
# https://classroom.udacity.com/courses/ud810/lessons/2952438768/concepts/29548388720923
# ----
# Correction to the rotation matrix that is shown. Instead use Wiki
# https://en.wikipedia.org/wiki/Rotation_matrix
# theory can also be foudn in my Power

# To switch the translation and rotation input to be relative to the Auotware
#   coord sys multiply inputs by -1
#   This is not how Camera transformations work.  But is more logical and 
#   helps when transitioning between coord sys because you only need to 
#   provide inputs based on 1 system.

# CAUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
# Rotation is applied in the order of x,y,z
# Rotation is applied 1st then translation

# INPUTS 
# Pa - Point in world frame
# gamma - rotation about x axis
#          - CameraTranslation: what rotation of the camera would get it back to world frame
#          - Autoware/Unity: rotation of unity measured in autoware (remember to change code to use factor of -1)
# kappa - roation about y axis
#          - CameraTranslation: what rotation of the camera would get it back to world frame
#          - Autoware/Unity: rotation of unity measured in autoware (remember to change code to use factor of -1)
# theta - rotation about z axis
#          - CameraTranslation: what rotation of the camera would get it back to world frame
#          - Autoware/Unity: rotation of unity measured in autoware (remember to change code to use factor of -1)
# BAO - translation of b farme
#          - CameraTranslation: what translation of the camera would get it back to world frame
#          - Autoware/Unity: translation of unity measured in autoware (remember to change code to use factor of -1)


import numpy as np
from matplotlib import pyplot as plt

#1x3
Pa=np.matrix([[1],
              [1],
              [1]])
print ("Pa - Point in Autoware - expressed using Autoware coord sys")
print (Pa)

# Rotation ********************
gamma = 0*np.pi*2/360
#print ("gamma - x-axis rotation of Unity coord sys about Unity coord sys - in degrees")#use this for standard camera translation
print ("gamma - x-axis rotation of Unity coord sys about Autoware coord sys - in degrees") #used for single coord sys inputs - not standard for Camera Transformation 
print (gamma/(np.pi*2/360))
gamma = gamma * -1  # used for single coord sys inputs - not standard for Camera Transformation 

#3x3
BARx=np.matrix([[1,0,0],
                [0,np.cos(gamma),-np.sin(gamma)],
                [0,np.sin(gamma),np.cos(gamma)]])
#print ("BARx - x-axis rotation matrix of Unity coord sys about Unity coord sys")
#print (BARx)

kappa = 90*np.pi*2/360 
#print ("kappa - y-axis rotation of Unity coord sys about Unity coord sys - in degrees")#use this for standard camera translation
print ("kappa - y-axis rotation of Unity coord sys about Autoware coord sys - in degrees") #used for single coord sys inputs - not standard for Camera Transformation 
print (kappa/(np.pi*2/360))
kappa = kappa * -1 # used for single coord sys inputs - not standard for Camera Transformation 

#3x3
BARy=np.matrix([[np.cos(kappa),0,np.sin(kappa)],
                [0,1,0],
                [-np.sin(kappa),0,np.cos(kappa)]])
#print ("BARy - y-axis rotation matrix of Unity coord sys about Unity coord sys")
#print (BARy)

theta = 0*np.pi*2/360
#print ("theta - z-axis rotation of Unity coord sys about Unity coord sys - in degrees")#use this for standard camera translation
print ("theta - z-axis rotation of Unity coord sys about Autoware coord sys - in degrees") #used for single coord sys inputs - not standard for Camera Transformation 
print (theta/(np.pi*2/360))
theta = theta * -1 # used for single coord sys inputs - not standard for Camera Transformation 

#3x3
BARz=np.matrix([[np.cos(theta),-np.sin(theta),0],
                [np.sin(theta), np.cos(theta),0],
                [0,0,1]])
#print ("BARz - z-axis rotation matrix of Unity coord sys about Unity coord sys")
#print (BARz)

# Translation *****************
# Translation is applied after the rotation occurs
#1x3
BAO = np.matrix([[0],
                 [0],
                 [0]])
#print ("BAO - Tranlation of coord sys expressed in Unity coord sys") #use this for standard camera translation
print ("BAO - Tranlation of coord sys expressed in Autoware coord sys") #used for single coord sys inputs - not standard for Camera Transformation 
print (BAO)
BAO = BAO * -1 # used for single coord sys inputs - not standard for Camera Transformation

# Calculation **************************
# 3x3
BAR=BARx * BARy * BARz
#print ("BAR - rotation matrix of Unity coord sys about Autoware coord sys")
#print (BAR)

#1x3
ZeroT = np.matrix([[0,0,0]])
#print ("ZeroT - 1x3 zero matrix to make the solution homogeneous and invertable")
#print (ZeroT)

#4x4
RO = np.concatenate((BAR,BAO),axis=1)
One = np.matrix([[1]])
H = np.concatenate((ZeroT,One),axis=1)
BAT = np.concatenate((RO,H),axis=0)
#print ("BAT- Total tranlational and rotation matrix of Unity coord sys relative Autoware coord sys")
#print (BAT)

#4x1
PaH = np.concatenate((Pa,One),axis=0)
#print ("PaH - Adding 1 to the P x,y,z to make the solution homogeneous and invertable")
#print (PaH)
#print (np.shape(PaH))

#1x4
PbH = BAT * PaH
#print ("Pb - Point - expressed using Unity coord sys from homogeneous calulation")
#print (PbH)

#1x3
Pb = np.delete(PbH,3,0)
print ("Pb - Point - expressed using Unity coord sys")
print (Pb)

# Reverse should be true ******************
#1x4
PaHcheck = np.linalg.inv(BAT) * PbH
#print ("PaHcheck - Point - expressed using Autoware coord sys by inverting the Pb answer, homogeneous calulation")
#print (PaHcheck)



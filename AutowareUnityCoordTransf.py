# Run this in a command prompt or use Anaconda
# C:\Apps\Anaconda3\python.exe W:\PythonScripts\AutowareUnityCoordTransf.py
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:42:51 2019

@author: Todd Nelson
"""

import numpy as np
from matplotlib import pyplot as plt

Paw=np.matrix([[0],
               [0],
               [0]])
print ("Point in Autoware - expressed using Autoware coord sys")
print (Paw)

#***************************************************************


Baw=np.matrix([[30],
              [-7],
              [2]])
print ("Loction of Unity world expressed in Autoware world coordinates")
print (Baw)



Trhlh=np.matrix([[0,1,0],
                 [0,0,1],
                 [1,0,0]])
print ("Right hand (Autoware) to Left hand (Unity) Tranformation")
print (Trhlh)



Tawun=np.matrix([[-1,1],
                 [1,-1],
                 [1,-1]])
print("Autoware to Unity world transformation")
print (Tawun)



K = np.matrix([[Paw.item(0), Baw.item(0)],
               [Paw.item(1), Baw.item(1)],
               [Paw.item(2), Baw.item(2)]])
#print ("K")
#print (K)

M = Trhlh * K
#print ("M")
#print (M)

MT = M.transpose()
#print ("MT")
#print (MT)

PunFull = Tawun * MT
#print ("PunFull")
#print (PunFull)

PunX=PunFull[0,0]
PunY=PunFull[1,1]
PunZ=PunFull[2,2]
#print ("PunX", "PunY", "PunZ")
#print (PunX, PunY, PunZ)

Pun = np.matrix([[PunX],
                 [PunY],
                 [PunZ]])
print ("Point in Unity - using Unity coord sys")
print (Pun)




# ----------------------------------------------------------------------------
# Title: 
# Description: This script is used to process raw Interfermetry Image.This interpoaltes the small amount of nan in the image. This is required to caclute the volume of the colony
# Author: Aawaz R Pokhrel
# Date: November 9, 2023
# Citation: Pokhrel. et al. "The Biophysical basis for of bacterial colony growth" (2023)
# -------------------------------------------------------------------------------





import numpy as np
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy
import pandas as pd
from copy import  deepcopy
#from interpolate5X import*
from findCenter import findEstimatedcenter

'''This function takes image and return volume radius and hieght of the biofilm'''

def getInterpolatedImage(image, latRes,dataName,rootFolder,saveImageFolder):
    threshold=1
    xCenter,yCenter,estimatedRadius=findEstimatedcenter(image,threshold)  ##finds center throught differnet threhold method
    centerP=np.array([xCenter,yCenter])
    
    plotEstimatedCenter(image,xCenter,yCenter,saveImageFolder,dataName)
  
    print("Interpolating Image...")
    ##This is for dataset taken from 50X
    if (latRes<500):
        print ("SMALLER")
        image=interpolateImage(image,threshold,centerP,estimatedRadius,dataName,saveImageFolder) ## Threshold here is used to neglect the values  in volume calculation below certain height
        #@asd
    ##This is for dataset taken from 5X
    #if (latRes>500):
        #image=interpolateImage5X(image,threshold,centerP,estimatedRadius)
    plotFinalCalImage(image,dataName,saveImageFolder)
    #np.savetxt('/media/aawaz/Extra/interferometryBackup/july26/data/imageTrial.txt', image, delimiter=' ')
    return image

   


'''
This is the main function that takes the image and performs interoplation
image=2d array of image
threshold=
center
estimatedRadius

'''
def interpolateImage(image,threshold,center,estimatedRadius,dataName,rootFolder):


    image=image/1000
    interImage=interpolate(image,center,threshold,estimatedRadius)
    coarNan=np.where(np.isnan(interImage))
    plotRemainingNan(interImage,dataName,rootFolder)
    
    
    ##find the location fo remaining Nan's, you can check visually if there is remining Nan's left inside the bioiflm
    ## if No remaining Nan's inside the biofilm , turn every other Nan to zero as they dont matter in other calcuation
    interImage[coarNan] = 0
    bool1=np.any(np.isnan(interImage))
    print ("Remaining Nan's in the Image =  " +str(np.sum(bool1)))
    
    return interImage
 

'''This funtion interpolates a single profile'''
def interpolateProfile(data):
  #  plt.plot(data)
    s = pd.Series(data)
    try:
      s=s.interpolate(method='spline', order=1,limit_direction='both')
      data=scipy.signal.savgol_filter(s, 41, 1) ##you can change the width of the  inteprolation profile
      #plt.plot(s)
     # plt.show()
     # asd
    except ValueError as e:
      print (data)
      
      plt.plot(data)
      plt.show()
      asd
    
    
    return data

'''This function performs circular interpolation by first finding the center of bioiflm, fitting a circle to find a radius , using that radius to do circular interoplation
This is the main interpolation
'''
def interpolate(image,center,threshold,R):

    points = []
    sh=image.shape
    
    rn,cn=center[0],center[1]
    
    start=0
    end=360
    num=5000   #####3change this number for better/worse interpolation
    angles = np.arange(start, end, (end-start)/num)         ##Array for angles o go aorund the cirrale for interpolation
    profiles=[]
    radius=R+300
    lines=image
    newLines=deepcopy(lines)
    for counter,i  in enumerate(angles):
        units = np.arange(0, radius, 1) ##0
        pointsX=rn+units*np.cos(i * math.pi/180)
        pointsY=cn+units*np.sin(i * math.pi/180)
        pointsX=np.asarray(pointsX)
        pointsY=np.asarray(pointsY)
        pointsX=pointsX.astype(int)
        pointsY=pointsY.astype(int)
        booi=pointsX<sh[1]
        booi1=pointsY<sh[0]
        booi3=np.logical_and(booi, booi1)
        booi=pointsX>0
        booi1=pointsY>0
        booi4=np.logical_and(booi, booi1)
        booi5=np.logical_and(booi4, booi3)   
        pointsX=pointsX[booi5]
        pointsY=pointsY[booi5]
        #plt.plot(pointsX,pointsY) #C
        lCenP=lines[pointsY,pointsX]
        newLines[pointsY,pointsX]=interpolateProfile(lCenP)  #This is where the real interpolation for each profile takes place
      
        
    return newLines


def plotRemainingNan(interImage,dataName,rootFolder):
    savePath=rootFolder+'checkImages/'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    coarNan=np.where(np.isnan(interImage))
    plt.imshow(interImage,vmin=0,vmax=200)
    plt.title(dataName+' = Check for Nans inside biofilm')
   
    plt.scatter(coarNan[1],coarNan[0]) ##Plot values of nan
    plt.savefig(savePath+dataName+'nan.png')
  
    plt.close()
  
 
def plotRealCenter(image,xc,yc,dataName,finalFolder):
    plt.imshow(image,vmin=-10,vmax=100)
    plt.scatter(yc,xc)
    plt.title(dataName+" Check for Real center")
    plt.savefig(finalFolder+dataName)
    plt.show()
    

def plotEstimatedCenter(image,xc,yc,rootFolder,dataName):
  #  print (finalFolder)
    
    plt.imshow(image,vmin=-10,vmax=60)
    plt.scatter(xc,yc)
    plt.title("Check for Estimated center")
    savePath=rootFolder+'/checkImages/'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
   # plt.show()
  #  asd
    plt.savefig(savePath+dataName+'center.png')
    plt.close()
 #  plt.close()

    
def plotFinalCalImage(image,dataName,rootFolder):
    savePath=rootFolder+'checkImages/'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    plt.imshow(image,vmin=-10,vmax=50)
    #plt.scatter(xc,yc)
    plt.title(dataName+" Final Image")
    plt.savefig(savePath+dataName+'final.png')
    plt.close()



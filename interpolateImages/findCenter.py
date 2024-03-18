

# ----------------------------------------------------------------------------
# Title: 
# Description: This script is used to process raw Interfermetry Image.This interpoaltes the small amount of nan in the image. This is required to caclute the volume of the colony
# Author: Aawaz R Pokhrel
# Date: November 9, 2023
# Citation: Pokhrel. et al. "The Biophysical basis for of bacterial colony growth" (2023)
# -------------------------------------------------------------------------------




import numpy as np
import scipy
import pandas as pd
import math
from least_square_circle import*


    
'''This is the main function that takes the image and finds the center
Uses the threhold of 20 um for the first one and 5 um for the second one to find the least swaure circle and find the center
'''
def findEstimatedcenter(image,threshold):
   
    xCenter,yCenter,estimatedRadius=findCenter(image,threshold)
    return xCenter,yCenter,estimatedRadius
  
'''
This funcyion finds  center twice, to be more precise, with two differnet threshols
'''
def findCenter(lines,threshold):
    #plt.imshow(lines/1000,vmin=2,vmax=100)
    xCoar,yCoar=findPoints(lines)  #Usese 10 um threhold for the first time
    #plt.scatter(xCoar,yCoar)
    #plt.show()
   # asd
    xc,yc,R,residu=leastsq_circle(xCoar,yCoar)
    xCoar,yCoar=findPointsAgain(lines,threshold,xc,yc)   ##Uses around 2 um threhold
   # plt.scatter(xCoar,yCoar)
   # plt.show()
  #  asd
    xc,yc,R,residu=leastsq_circle(xCoar,yCoar)
    return int(xc),int(yc),R


'''This function find points around that lies near the 10 um threhold '''

def findPoints(image):
    image=image/1000

    thres=10   #3#threhold used in 10 um
   

    ##This is to interpolate the line profile to find the threhold
    def interpolateInner(xT,data):
        s = pd.Series(data)
        s=s.interpolate(method='spline', order=1,s=0,limit_direction='both')
        try:
            data=scipy.signal.savgol_filter(s, 41, 1)#,deriv=1,delta=5)   ##3UIse 41 vaue fo rinterpolateee
        except ValueError as e:
            plt.plot(xT,data)
            plt.show()
            ad
    

        return data




    '''This is to find the point where the height matches the threshold profile'''
    def findThres(profile):
    
      
        xI=np.arange(len(profile))
        #plt.plot(xI,profile)
        if(len(profile)==0):
            return np.nan
      #  profile=interpolateInner(xI,profile)
        
        indL=np.where(np.abs(profile-thres)<3)   ##Threshold 5 used for everything
        if(len(indL[0])>0):
           # plt.plot(xI,profile)
          #  plt.show()
            #asd
         #   print (profile)
            indi=indL[0][0]
            return indi,profile
        return np.nan,np.nan
    sh=image.shape
    print (sh)
  
    xc,yc=(sh[1]/2,sh[0]/2)
 
    num=4000  ##number aorund the circle
    start=0
    end=360
   
    angles = np.arange(start, end, (end-start)/num) 
    profiles=[]
    print (sh[0],sh[1])
    maxSize=np.max(np.array([sh[0],sh[1]]))
  
    radius=maxSize+1500
    rn=int(xc)
    cn=int(yc)
    xCoar=[]
    yCoar=[]
    #print (image)
 #   plt.imshow(image,cmap='viridis')
    for counter,i  in enumerate(angles):
        #print (i)
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
        lCenP=image[pointsY,pointsX]
       
        indThres,newProfile=findThres(lCenP)
        
        
        if(~np.isnan(indThres)):
                xCoar.append(pointsX[indThres])
                yCoar.append(pointsY[indThres])
      
    return  xCoar,yCoar

'''This function find points around that lies near the threhold 2 um '''

def findPointsAgain(image,thres,xc,yc):
    image=image/1000
    #image[image<0]=0
    print (thres)
    thres=2 #3Use 5 usually
 

    def interpolateInner(xT,data):
        s = pd.Series(data)
        s=s.interpolate(method='spline', order=1,s=0,limit_direction='both')
        try:
            data=scipy.signal.savgol_filter(s, 41, 1)#,deriv=1,delta=5)   ##3UIse 41 vaue fo rinterpolateee
        except ValueError as e:
            plt.plot(xT,data)
            plt.show()
            ad
    

        return data




    '''This is to find the point where the height matches the threshold profile'''
    def findThres(profile):
    
      
        xI=np.arange(len(profile))
        #plt.plot(xI,profile)
        if(len(profile)==0):
            return np.nan
      #  profile=interpolateInner(xI,profile)
        
        indL=np.where(np.abs(profile-thres)<1)   ##Threshold 5 used for everything
        if(len(indL[0])>0):
           # plt.plot(xI,profile)
          #  plt.show()
            #asd
         #   print (profile)
            indi=indL[0][0]
            #plt.axvline(x=indi)
           # profile[indi:]=0.0
           # plt.plot(xI[indi:],profile[indi:])
           # print(profile)
           # plt.plot(xI,profile)
           # plt.show()
           # ddd
       #     print (profile)
          #  print (indL)
            return indi,profile
        return np.nan,np.nan
    sh=image.shape
    print (sh)

    num=400
    start=0
    end=360
   
    angles = np.arange(start, end, (end-start)/num) 
    profiles=[]
    print (sh[0],sh[1])
    maxSize=np.max(np.array([sh[0],sh[1]]))
  
    radius=maxSize+1500
    rn=int(xc)
    cn=int(yc)
    xCoar=[]
    yCoar=[]
    #print (image)
 #   plt.imshow(image,cmap='viridis')
    for counter,i  in enumerate(angles):
        #print (i)
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
        lCenP=image[pointsY,pointsX]
    
        indThres,newProfile=findThres(lCenP)
       
    
        
        if(~np.isnan(indThres)):
                xCoar.append(pointsX[indThres])
                yCoar.append(pointsY[indThres])
       # profiles.append(lCenP)
    

    return  xCoar,yCoar






# ----------------------------------------------------------------------------
# Title: 
# Description: This script is used to process raw Interfermetry Image and subtract a backgorund for time lapse images.
# Author: Aawaz R Pokhrel
# Date: November 9, 2023
# Citation: Pokhrel. et al. "The Biophysical basis for of bacterial colony growth" (2023)
# -------------------------------------------------------------------------------
-------------------------------------------------------------------------

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy
import pandas as pd
from pylab import *
from skimage import filters
from least_square_circle import*


'''This function subtracts plane from the given image txt  file with latRes input from interferometry
findCircle= This finds the edge of bioiflm by finding gradient of the image and threholdinog that gradient
Note here that a lot of these functions involve finding the center of biofilm, going aorund the circle and then finding
the regions outise the colony. We then find the agar portion , subtract a best fit polynoimial and return the final image

'''
    
def subtractPlane(image, latRes):
    plotBestFitCircle=True  ##This plots the best fit circle for the viewer to see and judge by themselves
    xCoar,yCoar=findCircle(image,latRes)   ##This find the circle around the colony witht some threshold
    xc,yc,R,residu=leastsq_circle(xCoar,yCoar)   
    x,y,z=findRegionOutside(image,xc,yc,R)   ##this is to find the region outside the colony to find the agar surface
    if(plotBestFitCircle):
        plotBestCircle(image,xc,yc,R)           
    plotSubRegion(x,y,z)
    imageSub=subPlane(x,y,z,image,latRes)   ##This uses the region outisde to fit a polynimila and subtract
    plotSubtractedImage(imageSub)               ##plot if necessary
    return imageSub                 


def  findCircle(image,latRes):
   
    thresGrad=60 #60 UIsed
    if (latRes>1000): ## This is for image woth 5.X
          thresGrad=20
    image2=np.gradient(image)[1]
   
    imageGauss= filters.gaussian(image2, sigma=3)

 
    imageGauss=interpolate(imageGauss)
 
    xCoar,yCoar=findCircleCoardCol(np.absolute(imageGauss),thresGrad)
    plotCoar(imageGauss,xCoar,yCoar)  ##this plots the coardinates of the points to fit the best fit circle 
    return xCoar,yCoar



'''
Inteprolates the data, by basically filling up the nan's
'''
def interpolate(lines):
        df = pd.DataFrame(lines)
        df.fillna(method='ffill', axis=1, inplace=True)
        df.fillna(method='bfill',axis=0,inplace=True)
        return df.values


'''
This function goes columns by Columns to see where the first gradient reaches the threhold value
This works better than doing  row by row
'''

def findCircleCoardCol(image,thres):
    image[(image>100)]=0
    image[np.isnan(image)]=0
   
    '''This is to find the point where the gradient matches the threshold profile'''
    def findThres(profile):
      
        xI=np.arange(len(profile))
        if(len(profile)==0):
            return np.nan
        indL=np.where(np.absolute((profile-thres))<5)   ##Threshold 5 used for everything
        if(len(indL[0])>0):
            
            ind1=indL[0][0]
            ind2=indL[0][-1]
            return ind1,ind2,profile
        return np.nan,np.nan,np.nan
   
    xCoar=[]
    yCoar=[]
    sh=image.shape
    print (sh[1])
    rangeN=np.arange(0,sh[0],10)
  
    '''
    This goes row by row to find the profile and then finds the thresholdS
    '''

    for counter,i  in enumerate(rangeN):
        pointsX=np.repeat(i, sh[1])
        pointsY=np.arange(0,sh[1])
        lCenP=image[pointsX,pointsY]
        indThres1,indThres2,newProfile=findThres(lCenP)
       
        if(~np.isnan(indThres1)):
           # yCoar.append(i)  ## remove comment  for half data
            yCoar.append(i)
           # xCoar.append(pointsY[indThres1])## remove comment for half data
            xCoar.append(pointsY[indThres2])
   
    return xCoar,yCoar


 
    
def subPlane(x,y,z,image,latRes):
     
        ###This returns the best fit plane fropm the given points
        def returnBestPlane(x,y,z):
            x=x*latRes
            y=y*latRes
            
            tmp_A = []
            tmp_b = []
            for i in range(len(x)):
                tmp_A.append([x[i], y[i], 1])
                tmp_b.append(z[i])
            b = np.matrix(tmp_b).T
            A = np.matrix(tmp_A)
           #print (A)
           # asd
            fit, residual, rnk, s = lstsq(A, b)
            return fit[0], fit[1], fit[2]



        #ax.scatter(x, y, z, color="r",s=0.02)
        a,b,c=returnBestPlane(x,y,z)
        X=np.arange(0,image.shape[0])*latRes
        Y=np.arange(0,image.shape[1])*latRes
        xW,yW = np.meshgrid(X, Y)
        xF=xW.flatten()#[booi]
        yF=yW.flatten()#[booi]
       # print (dataM.shape)
        zF=image.T.flatten()#[booi]\
      #  print (np.nanmax(zF))
      #  print(zF)
        zFit=a*xF+b*yF+c
      #  print(zFit)
     #   print (zFit.shape)
        #asd
        finalZ=zF-zFit              ##Subtarct best fit plane
      #  print (finalZ)
        #sd
       
        finalZ=finalZ.reshape((image.shape[1],image.shape[0]))
        finalZ=finalZ.T
        return finalZ    
    
'''
This function find points around that lies near the threhold provided


'''

def findRegionOutside(image,cx,cy,radii):
   # plt.imshow(zImage)
   # plt.show()
    #asd


    def interpolateInner(xT,data):
        s = pd.Series(data)
        s=s.interpolate(method='spline', order=1,s=0,limit_direction='both')
        try:
            data=scipy.signal.savgol_filter(s, 81, 1)#,deriv=1,delta=5)  
        except ValueError as e:
            plt.plot(xT,data)
            plt.show()
            ad
    

        return data




    '''This is to find the point where the height matches the threshold profile'''
    def findThres(profile,thres):
    
      
        xI=np.arange(len(profile))
       # plt.plot(xI,profile)
        if(len(profile)==0):
            return np.nan
        profile=interpolateInner(xI,profile)
        
        indL=np.where(np.abs(profile)>thres)   ##Threshold 5 used for everything
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
            #plt.show()
           # ddd
       #     print (profile)
          #  print (indL)
            return indi,profile
        return np.nan,np.nan
    sh=image.shape
 #   plt.imshow(image)
    xc,yc=cx,cy
  #  print (xc,yc)
  #  plt.scatter(xc,yc)
    num=1000
    start=0
    end=360
   
   
    angles = np.arange(start, end, (end-start)/num) 
    profiles=[]
    radius=radii+100
    extra=200
    rn=int(xc)
    cn=int(yc)
    xCoar=[]
    yCoar=[]
    print (image)
    x=[]
    y=[]
    z=[]
   # plt.imshow(image,cmap='viridis')
    for counter,i  in enumerate(angles):
        #print (i)
        units = np.arange(radius, radius+extra, 1) ##0
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
       # plt.plot(pointsX,pointsY)
        
        #print (pointsX)
        ##asd
        x.append(pointsY)
        y.append(pointsX)
        lCenP=image[pointsY,pointsX]
        z.append(lCenP)
      
       
    
    #plt.show()
    #asd
    x=np.array(x)
    x=np.concatenate( x, axis=0 )
    y=np.concatenate( y, axis=0 )
    z=np.concatenate( z, axis=0 )
    #Remove nans
    booli=np.isnan(z)
    x=x[~booli]
    y=y[~booli]
    z=z[~booli]

    
   #Remove outliers
    m=2
    boolTrim=abs(z - np.mean(z)) < m * np.std(z)
    
    z=z[boolTrim]
    x=x[boolTrim]
    y=y[boolTrim]

    return  x,y,z

'''This function goes row by row to see where the first gradient reaches the threhold value'''
def findCircleCoardRow(image,thres):
    image[(image>100)]=0
    image[np.isnan(image)]=0
   
    '''This is to find the point where the gradient matches the threshold profile'''
    def findThres(profile):
      
        xI=np.arange(len(profile))
       #plt.plot(xI,profile)
        if(len(profile)==0):
            return np.nan
       # profile=interpolateInner(xI,profile)
       # print (profile)
        #asd
        indL=np.where(np.absolute((profile-thres))<5)   ##Threshold 5 used for everything
        if(len(indL[0])>0):
       
            indi=indL[0][0]
            return indi,profile
        return np.nan,np.nan
   
    xCoar=[]
    yCoar=[]
    sh=image.shape
   # print (sh[1])
    rangeN=np.arange(0,sh[1],5)
   # print (rangeN)
  
    for counter,i  in enumerate(rangeN):
        pointsX=np.repeat(i, sh[0])
        pointsY=np.arange(0,sh[0])
      #  print (pointsX,pointsY)
        lCenP=image[pointsY,pointsX]
        indThres,newProfile=findThres(lCenP)
        print (indThres)
        
        if(~np.isnan(indThres)):
            xCoar.append(i)
            yCoar.append(pointsY[indThres])
    return xCoar,yCoar
  
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
'''
def interpolateCircular(image,center,threshold,R):
#    image=image/1000
   # plt.imshow(image)
   # plt.show()
   # asd
    points = []
    sh=image.shape
    
    rn,cn=center[0],center[1]
    
#    xCoar,yCoar=findPoints(image,center,threshold)  ##find points of just the biofilm
    #plt.imshow(image)
    #plt.scatter(xCoar,yCoar)
    
  #  plt.show()
  #  asd
#    xc,yc,R,residu=leastsq_circle(xCoar,yCoar)  ##find a best fit circle that fits there to calcualte the radius 
 #   plt.scatter(rn,cn)
   # plt.show()
    #asd
    start=0
    end=360
    num=50000
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
       # plt.plot(lCenP)
      #  plt.show()
       # profiles.append(lCenP)
       # print(pointsX,pointsY)
       # plt.plot(pointsX,pointsY)
    #plt.show()
        #sad
        
    return newLines


'''
Following functions are to plot and see if it's fitting the right plane
'''

def plotBestCircle(image,xc,yc,R):
    image2=np.gradient(image)[1]
    imageGauss= filters.gaussian(image2, sigma=3)
    theta_fit = np.linspace(-math.pi,math. pi, 180)
    x_fit = xc + R*np.cos(theta_fit)
    y_fit = yc + R*np.sin(theta_fit)
    plt.imshow(np.absolute(imageGauss),vmin=-5,vmax=200)
    plt.scatter(xc,yc,s=500)
    plt.plot(x_fit, y_fit, 'r' , label="fitted circle", lw=2)
    plt.title('Least Squares Circle')
    plt.show()
    
   
def plotSubtractedImage(image):
    plt.imshow(image/1000,vmin=-5,vmax=120)
    plt.title("Final Background Subtracted Image")
    plt.show()


def plotSubRegion(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color="r",s=0.02)
   # ax.scatter(xF, yF, zFit, color="b",s=0.02)
   # ax.scatter(xF, yF, finalZ, color="g",s=0.02)
    ax.set_title("Region used to get fit plane")
    plt.show()
   # asd
    

def plotCoar(imageGauss,xCoar,yCoar):
    plt.imshow(imageGauss,vmin=-1  ,vmax=300)
    plt.title("Circle coardinates Detected by algorithm")
    plt.scatter(xCoar,yCoar,s=1)
    plt.show()
    
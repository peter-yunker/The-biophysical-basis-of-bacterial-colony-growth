# ----------------------------------------------------------------------------
# Title: 
# Description: This script is used to process raw Interfermetry Image.This interpoaltes the small amount of nan in the image. This is required to caclute the volume of the colony
# Author: Aawaz R Pokhrel
# Date: November 9, 2023
# Citation: Pokhrel. et al. "The Biophysical basis for of bacterial colony growth" (2023)
# -------------------------------------------------------------------------------

import numpy as np
import os
from interpolateFunctions import getInterpolatedImage


'''Reads 2d array from image file'''
def getImage(file):
  #  print (file)
 #   asd
    try:
        lines =np.loadtxt(file, delimiter='\t')
    except ValueError as e:
         lines =np.loadtxt(file, delimiter=' ')
    return lines

'''
This function takes an input of folder where the data are are subtracts the background
'''

def interpolateImage(filePath,saveImageFolder,destFolder):
    latCat=173.6296296296   ##read the latResoultion or supply it directly
    blockReduce=2
    latRes=latCat*blockReduce
    dataName=os.path.basename(os.path.normpath(filePath))[:-13] #this just gets the apporariate  get dataName. change accordingly 
   
    finalDataName=dataName+'(nointerFinal).txt' ###
    image=getImage(filePath)
    #plt.imshow(image)
    #plt.show()
    #asd

    finalImage=getInterpolatedImage(image,latRes,dataName,filePath,saveImageFolder) ##The image should be in 2d array format
   # plt.imshow(finalImage)
   # plt.show()
   # asd
    
    np.savetxt(destFolder+finalDataName, finalImage, delimiter='\t',fmt='%3f')
   

    
'''
Input to the functions here
'''


''' A sample on how to run the code 
folderName=getPath()   ##Enter the filePath where the datx file is located
destFolderName=getPath()   ##Enter the destPath where the txt file made after processing

folderPath=getPath()+'sampleData/rawImage/'  ##Where file is located
destFolderName=getPath()+'sampleData/rawImage/'  ##3 Where 


filePath=folderPath+'data-1(nointerFinal).txt'
saveImageFolder=folderPath  #This is where to save the interoaltion chekc images while it's intepolating


interpolateImage(filePath,saveImageFolder,destFolderName)  ##supply the file Path here, where to save the images (images to see if it's correctly interpolated), and folder on where to saeve the interpoalted image
'''
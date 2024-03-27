# ----------------------------------------------------------------------------
# Title: 
# Description: This script is used to process raw Interfermetry Image and subtract a backgorund for time lapse images.
# Author: Aawaz R Pokhrel
# Date: November 9, 2023
# Citation: Pokhrel. et al. "The Biophysical basis for of bacterial colony growth" (2023)
# -------------------------------------------------------------------------------
#-------------------------------------------------------------------------

import numpy as np
import os
from subPlaneFinal import subtractPlane


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

def subtractBackground(filePath,destFolder):
    latCat=173.6296296296   ##read the latResoultion or supply it directly ##Thic values is for 50X lens
    blockReduce=2    #block reduce to reduce the sampling of data #The data are usually downsmapled by factor of 2
    latRes=latCat*blockReduce
    dataName=os.path.basename(os.path.normpath(filePath))[:-13] #this just gets the apporariate  get dataName. chage accordingly 
   
    finalDataName=dataName+'(nointerFinal).txt'
    image=getImage(filePath)
    finalImage=subtractPlane(image,latRes)
    
    np.savetxt(destFolder+finalDataName, finalImage, delimiter='\t',fmt='%3f')
   
    
'''
Input to the functions here
'''

####Sample  Runm
#folderName=getPath()   ##Enter the filePath where the datx file is located
#destFolderName=getPath()   ##Enter the destPath where the txt file made after processing

#folderPath=getPath()+'sampleData/rawImage/'
#destFolderName=getPath()+'sampleData/rawImage/'


#filePath=folderPath+'data-1(nointer).txt'

#subtractBackground(filePath,destFolderName)  ##supply the file Path here Run this with file path
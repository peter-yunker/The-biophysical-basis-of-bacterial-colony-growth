# ----------------------------------------------------------------------------
# Title: 
# Description: This script is used to process raw Interfermetry Image. It convert the datx file to image j and python redeable txt image.
# Author: Aawaz R Pokhrel
# Date: November 9, 2023
# Citation: Pokhrel. et al. "The Biophysical basis for of bacterial colony growth" (2023)
# -------------------------------------------------------------------------------
#-------------------------------------------------------------------------
import numpy as np
import glob
import pandas as pd
import os
import math
import h5py
from skimage.measure import block_reduce
from readFunc import getZ



'''
Input to the functions here
'''

folderName= 'sampleData/rawImage/' ##Enter the filePath where the datx file is located
destFolderName='sampleData/rawImage/final' ##Enter the destPath where the txt file should be  made after processing
blockReduce=2               ##This is to blokcRedudce large file by taking mean of the block size (See skimage.measure.blokcReduce)

saveImageraw(folderName,destFolderName,blockReduce)





'''
This functions gets all the datx file in a given folder and converts them into txt file
Input: inputFolder, destFolder, blockSize for image reduction
'''
def saveImageraw(mainFolder,destFolder,blockSize):
    

    latFolder=mainFolder+'latCat.txt' ##This will save a txt file with lateral REsolution and blockReduce
    df = pd.DataFrame(columns = ['fileName','latRes','blockReduce'])  #3this is to save the block Reduce factor and lateral resolution

    filesMain=glob.glob(mainFolder+"*.datx")

    
    counter=0
    for counter1,finalFile in enumerate(filesMain):
            dataName=os.path.basename(os.path.normpath(finalFile))[:-5] #this just takes out datx to get dataName
            fileName=dataName+'.txt'                           ## Orig fileName+destFileame 
            
            finalFolder=destFolder+fileName
        
            zdata,latcat=getZ(finalFile)                                #3this converts datx to txt
            if(not(math.isnan(latcat))):
                latCatNM=latcat*1E9
                print (latCatNM)
                data=[dataName,latCatNM,blockSize]
                zdata=np.around(zdata,decimals=2)  #3rounds the data to 2 decimal
                zdata=block_reduce(zdata, block_size=(blockSize,blockSize), func=np.nanmean) ##block reduces it
                df.loc[counter]=data
                counter=counter+1
                np.savetxt(finalFolder, zdata, delimiter=' ') #3save the image file
    df.to_csv(latFolder)  #save the lateral reolustion file
   



'''
These sets of from here code reads the datx interferometry file to convert to txt

'''
def datx2py(file_name):
    """Loads a .datx into Python, credit goes to gkaplan.
    https://gist.github.com/g-s-k/ccffb1e84df065a690e554f4b40cfd3a"""
    def _group2dict(obj):
        return {k: _decode_h5(v) for k, v in zip(obj.keys(), obj.values())}
    def _struct2dict(obj):
        names = obj.dtype.names
        return [dict(zip(names, _decode_h5(record))) for record in obj]
    def _decode_h5(obj):
        if isinstance(obj, h5py.Group):
            d = _group2dict(obj)
            if len(obj.attrs):
                d['attrs'] = _decode_h5(obj.attrs)
            return d
        elif isinstance(obj, h5py.AttributeManager):
            return _group2dict(obj)
        elif isinstance(obj, h5py.Dataset):
            d = {'attrs': _decode_h5(obj.attrs)}
            try:
                d['vals'] = obj[()]
            except (OSError, TypeError):
                pass
            return d
        elif isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.number) and obj.shape == (1,):
                return obj[0]
            elif obj.dtype == 'object':
                return _decode_h5([_decode_h5(o) for o in obj])
            elif np.issubdtype(obj.dtype, np.void):
                return _decode_h5(_struct2dict(obj))
            else:
                return obj
        elif isinstance(obj, np.void):
            return _decode_h5([_decode_h5(o) for o in obj])
        elif isinstance(obj, bytes):
            return obj.decode()
        elif isinstance(obj, list) or isinstance(obj, tuple):
            if len(obj) == 1:
                return obj[0]
            else:
                return obj
        else:
            return obj
    with h5py.File(file_name, 'r') as f:
        h5data = _decode_h5(f)
    return h5data


def zarray_datx(file, norm):
    """Returns a numpy array that is the z-values on each point"""
    myh5 = datx2py(file)
    zdata = myh5['Data']['Surface']
    print (zdata)
    zdata = list(zdata.values())[0]
    zvals = zdata['vals']
    zvals[zvals == zdata['attrs']['No Data']] = np.nan
    if norm:
        zvals = zvals-np.nanmean(zvals)
  #  data = subplane(zvals)
    print('Z-values are in: ', zdata['attrs']['Z Converter']['BaseUnit'])
    return zvals

"""Returns the zvalues and lateral resolution for the given file path"""
def getZ(fileName):
    h5data = datx2py(fileName)
    #print (h5data)
    #asd
    try:

        zdata = h5data['Data']['Surface']

    except KeyError:

        print("Data not found")
        return np.nan,np.nan
    
    zdata = list(zdata.values())[0]
    zvals = zdata['vals']
    zvals[zvals == zdata['attrs']['No Data']] = np.nan
    zunit = zdata['attrs']['Y Converter']['Parameters'][1]
    latcat=zunit
    return zvals,latcat

def latres(filepath):
    file = datx2py(filepath)
    k1 = list(file['Attributes'].keys())[1]
    k2 = list(file['Attributes'][k1]['attrs'].keys())[2]
    return file['Attributes'][k1]['attrs'][k2]




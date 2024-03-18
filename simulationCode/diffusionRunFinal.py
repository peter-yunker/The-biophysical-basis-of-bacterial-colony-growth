# ----------------------------------------------------------------------------
# Title: 
# Description: This script is used to geenrate height profiles by solving a swecond order differtial equation.
# Author: Aawaz R Pokhrel
# Date: November 9, 2023
# Citation: Pokhrel. et al. "The Biophysical basis for of bacterial colony growth" (2023)
# -------------------------------------------------------------------------------




import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from  diffusionFunction import diffusionRun


##Parameters

#D_updated = 100  # in μm²/hr
##Choose alpha,L,D and beta vlaues here
#alphaArray = np.array([1.113*2,1.113,1.113/1.5,1.113/2])
alphaArray = np.array([0.3,0.5,0.7,0.9,1.1])
betaArray = np.array([0.01,0.02,0.03,0.04])
DArray=np.array([20,50,100,200])
LArray=np.array([5,10,15,20,25,30])

#L = 7 # um
xFac=6000
counterNew=0
dfFinal = pd.DataFrame(columns=['alpha','beta','timeList','dataHeight','xArray','xFac','maxX','L','D'])
for L in LArray:
    for D_updated in DArray:
        for alpha in alphaArray:
            for beta in betaArray:
                maxX=3000  ##This is the x  range limit
                print ( "Param = ")
                print (L,alpha,beta,D_updated)
                df=diffusionRun(D_updated,alpha,beta,L,xFac,maxX)
                dataHeight=df.to_numpy()
                x=df.columns.values
        
                timeList=df.index.values
                data=[alpha,beta,timeList,dataHeight,x,xFac,maxX,L,D_updated]
        
                dfFinal.loc[counterNew]=data
                print (dfFinal)
                
                print(dfFinal)
                counterNew=counterNew+1
fileName='dataPickle/sim4/dataDiffuseFinal.pkl' ##Enter path to save the file here
print (fileName)
dfFinal.to_pickle(fileName)  #=this saves the simualated height profiles in pickle file
   

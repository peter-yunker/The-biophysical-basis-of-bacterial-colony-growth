# The biophysical basis of bacterial colony growth


 ----------------------------------------------------------------------------
 Title: 
 Description: These scripts are used for processing interferometry images used in the paper and to generate the simulated colonies discussed.
 Author: Aawaz R Pokhrel
 Date: November 9, 2023
 Citation: Pokhrel. et al. "The Biophysical basis for of bacterial colony growth" (2023)
-------------------------------------------------------------------------------
-------------------------------------------------------------------------


Background Subtraction: This folder contains codes for subtracting the background from the initial interferometry image. The dataset should include a sufficient agar surface so that it can detect the agar and fit a best-fit polynomial for subtraction

interpolateImages: This folder contains code to interpolate images with nan. Interpolation is necessary to calculate the volume of the colony. Splines interpolation is implemented. Since the data are radially symmetrical, we start from the center of the colony and draw lines pointing radially outward to interpolate

simulationCode: This folder contains code  to generate simulated height profiles.
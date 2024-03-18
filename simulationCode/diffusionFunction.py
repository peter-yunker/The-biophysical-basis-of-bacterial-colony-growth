
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd


def diffusionRun(D_updated,alpha,beta,L,xFac,maxX):


    x_extended = np.linspace(0, maxX, xFac)
    

    dx_extended = x_extended[1] - x_extended[0]

    #Initialise the Gaussian to emulate growth from single cells
    A = 1.2  # height of the Gaussian in um
    mu = maxX/2  # center of the Gaussian
    sigma = 1.5  # standard deviation (width) of the Gaussian in um

    h_initial_extended = A * np.exp(-((x_extended - mu)**2) / (2 * sigma**2))


    def pde_rhs_1d_extended(t, h):
        h_2d = h.reshape((-1, 1))
       # print (t)
        diffusion = np.zeros_like(h_2d)
    
    # Vectorized calculation of diffusion term
        diffusion[1:-1] = D_updated * (h_2d[:-2] - 2 * h_2d[1:-1] + h_2d[2:]) / dx_extended**2
    
    # Reaction term calculation using numpy's minimum and straightforward arithmetic
        reaction = alpha * np.minimum(h_2d, L) - beta * h_2d 
    
        return (diffusion + reaction).flatten()


    time_range_extended = [0, 50]  # 0 to 50 hours
    time_points_extended = np.arange(0, 50, 1)  # plotting every 10 hours
    solution_extended = solve_ivp(pde_rhs_1d_extended, time_range_extended, h_initial_extended.flatten(), method='RK45', t_eval=time_points_extended)


# Creating a DataFrame with height profiles
    height_profiles = solution_extended.y.T  # Transposed to get height profiles in rows
    df_height_profiles = pd.DataFrame(height_profiles, columns=x_extended, index=time_points_extended)
    return df_height_profiles
  




    

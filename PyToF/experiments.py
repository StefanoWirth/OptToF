import numpy as np
import os
from PyToF import ClassToF




kwargs = {}
N = 2**10
kwargs['N']      = 2**10
kwargs['G']      = 6.6743e-11
kwargs['M_init'] = 6836525.21*(1000)**3/kwargs['G'] #https://doi.org/10.1051/0004-6361/202244537
kwargs['R_init'] = 25225*1e3                        #https://iopscience.iop.org/article/10.1088/0004-6256/137/5/4322
kwargs['Period'] = 15.9663*60*60                    #https://www.sciencedirect.com/science/article/pii/S0019103511001783?via%3Dihub
kwargs['P0']     = 1e5

kwargs['Target_Js'] = [3401.655e-6, -33.294e-6] #https://doi.org/10.1051/0004-6361/202244537
kwargs['Sigma_Js']  = [   3.994e-6,  10.000e-6] #https://doi.org/10.1051/0004-6361/202244537

param = [3.21273277e+05, 2.86145345e+00] #https://arxiv.org/abs/2111.15494
def atmosphere(norm_radii, pressure, param=param):
    return (pressure/param[0])**(param[1]/(param[1]+1))

kwargs['use_atmosphere']   = True
kwargs['atmosphere_until'] = 1e7
kwargs['atmosphere']       = atmosphere

X = ClassToF.ToF(**kwargs)



""" #3 polytrope

N         = 2**10
density   = 1000 #kg/m^3
densities = density*np.ones(N) 
densities += np.linspace (0,1000,N) 
radius    = 1e6 #m
radii     = radius*np.logspace(0, -1, N)                 #note that all arrays start with the outermost part of the planet
mass      = -4*np.pi*np.trapezoid(densities*radii**2, radii) #negative sign because beginning of the array is the outer surface
period    = 24*60*60 #s

X = ClassToF.ToF(N=N, M_init=mass, R_init=radius, Period=period) """




def barotrope(P, param=None):
    rho = np.zeros_like(P)
    rho[:round(param[0]) ] = (P[:round(param[0]) ] / param[2])**(param[5] / (param[5] + 1))
    rho[ round(param[0]):] = (P[ round(param[0]):] / param[3])**(param[6] / (param[6] + 1))
    rho[ round(param[1]):] = (P[ round(param[1]):] / param[4])**(param[7] / (param[7] + 1))
    return rho

baro_param_init = np.array([N/2, N/2,       #transition radii indeces
                            2e5, 2e5, 2e5,  #bulk moduli
                            0.5, 0.5, 0.5,  #exponents
                            N/50])*2        #atmospheric index (optional)






#limits
def check_baro_param(param, give_atmosphere_index=False):

    if give_atmosphere_index:
        return param[-1]

    if   (param[0] < 0) or (param[0] > N) or (param[1]  < 0) or (param[1] > N):
        return True #transition radii out of bounds

    elif (param[2] < 0) or (param[3] < 0) or (param[4]  < 0):
        return True #negative bulk moduli

    elif (param[5] < 0) or (param[6] < 0) or (param[7]  < 0):
        return True #negative exponents

    elif (param[-1] < 0) or (param[-1] > param[0]):
        return True #atmosphere transition radius out of bounds

    else:
        return False


X.opts['baro_param_init'] = baro_param_init
X.set_barotrope(barotrope)
X.set_check_param(check_baro_param)
X.opts['verbosity'] = 2

X.opts['MaxIterHE'] = 100

print('Order of the Theory of Figures to be used:          ', X.opts['order'])
print('Initial mass of the planet in SI units:             ', X.opts['M_init'])
print('Initial equatorial radius of the planet in SI units:', X.opts['R_init'])
print('Initial rotation period of the planet in SI units:  ', X.opts['Period'])
print('Verbosity:                                          ', X.opts['verbosity'])
print('A few Wisdom & Hubbard 2016 Bessel values for J_n:  ', X.opts['Target_Js'])

nwalkers        = 20
steps           = 1000
Ncores          = 7
parallelize     = False

X.run_dens_opt(nwalkers, steps, Ncores=Ncores, parallelize=parallelize)




""" X.li         = radii
X.rhoi       = densities
X.m_rot_calc = (2*np.pi/period)**2*X.li[0]**3/(X.opts['G']*mass)
X.opts['MaxIterHE'] = 100 #Note that the default is only X.opts['MaxIterHE'] = 2, if relax_to_HE() is called directly, one should increase this number.
number_of_iterations = X.relax_to_HE()
print('Number of iterations used by the algorithm:', number_of_iterations)

X.plot_xy(0,2, dpi=75)

print('PyToF solutions:', ['J_'+str(2*i)  +' = '+"{:.4e}".format(X.Js[i]) for i in range(1,5)])

print(X.SS[2][-1])
print(X.SS[2][-2])
print(X.SS[3][-1])
print(X.SS[3][-2])
print(X.SS[4][-1])
print(X.SS[4][-2]) """
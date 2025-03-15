###########################################################
# Author of this version: Stefano Wirth - stwirth@ethz.ch #
###########################################################

"""
...............................................................................................................................          
..........:::::::::::::::::::::..::....:.......................................................................................          
..........::::::::::::::::::::::::::::::::::::::..:::::::::::::......:::::::::::::::::::::::::::::::::::::::::::::::::::.....--          
.........::::::::::::::::::::::::::::..::::::::::::::::::::::::......::::::::::::::::::::::::::::::::::::::::::-:::::--::....-+          
.........:::::::::::::::::::::-::::-=##+-::::::::::::::::::::::......:::::::::-::::::::::::::::::::::::::::--:--:::-:::::.....+          
........::::::::::::::::::-#%%%%%%%%**%%%##%%%-..::::::::::::::......:::::::-::::::::::::-::-=#%%#=::::::::-:-------:-:::.....+          
........::::::::::::::#%%%%%%%%%%%%#*=:.-+###%%%%#+:::::::::::::.....::::::::-:::::::=*%%%%%%%#==#%#+#%#+=--::----------:.....+          
........::::::::::::-%%%%%%%%%%%%%%%%##**++**####%%#-:::::::::::.....:::::::::::::+%%%%%%%%%%%#*=-:-=+**##%%=-----------::....-          
.......::::::::::::%%%%%%%%%%%%%%%%%%##*+===--=+*#%%%=::::::::::.....::::::::::::-%%%%%%%%%%%%%%##*+=====+*##%=----------:.....          
.......:::::::::::%%%%%%%%%%%%%%%%%###*+==----=+*#%%%%:::::::::......:::::::::::#%%%%%%%%%%%%%%#*++==--:--=+##%=---------:.....          
.......::::::::::=%%%%%%%%%%%%%%%%%%%%#+=-:::--=+##%%%#:::::::::.....:::::::::::%%%%%%%%%%%%%%##*+=--:::::=+*##%=---------:....          
.......::::::::::%%%%%%%%%%%%%%%####%%%#*+++#%%%%%%%%%-::::::::......:::::::::::%%%%%%%%%%%##@@@@@@@@@==@@@@@@@@@@:-------:....          
......::::::::::=%%%%%%%%%%%%%%%%%+=+%%%#==##%%#*##%%%-::::::::......::::::::::=%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=-:------....          
......::::::::::%%%%%%%%%%%%%########%%%%###++*+######::::::::::.....:::::::::=%%#%%%%%%%%#-@@@@@@@@@@@@@@@@@@@@@@+:-------....          
......::::::::::=%%%%%%%%%%%%#++==+%%%%%%**##=--=+=#%#::::::::::.....::::::::::-%%%%%%%%%%%%%@@@@@@@@@#++@@@@@@@@+---------:...          
.....:::::::::::-%%%%%%%%%%%%####%%%%%%%%##%#*+=--=*#=::::::::::.....:::::::::::=%%%%%%%%%%%#*+==#%%%%%+##*###+#+-----------...          
.....:::::::::::%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%#####::::::::::......::::::::::-%%%%%%%%%%%%%####%%%%%%%%%+==++#=-----------...          
.....::::::::::+%%%%%%%%%%%%%%%%%%*+++=*=+#%%%%%%%%#::::::::::::.....::-::::::::#%%%%%%%%%%%%%%%%%%%%%#****####=:::---------:..          
.....:::::::-+%%%%%%%%%%%%%%%%%%%%%%+=....:###%#%%#:::::::::::::.....::::::::-%%%%%%%%%%%%%%%%%%%%%%*==--=+#%%+-----------=-:..          
.....::::-#%%%%%%%%%%%%%%%%%%%%%%%%%%#+==-+#%%%%%:::::::::::::::.....::::::-%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-----------=-=--..          
.....::=%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%###*%%+:::::::::::::::::.....:::-#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%=-------====-=====..          
....+%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-::::::::::::::---::.....*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-----------========:.          
...=%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%:::::::--:::::-:::-:.....%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-------=--=======:.          
...*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%=::::::::::::--::::.....%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%=-------========.          
...%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*-:::::::::::::::.....%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#=---========.          
...%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*-:::::::::.....%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%+-=======:          
..-%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%=:::......%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%+-=====-          
..+%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#-......%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%=====-          
..#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%+.....%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%=====          
..===++++++++*++*******#**####**#**####*#######################=.....+#############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%====          
...............................................................................................................................          
 __  __             _        ____           _     _____     _____    _____ _                            _                             
|  \/  | ___  _ __ | |_ ___ / ___|__ _ _ __| | __|_   _|__ |  ___|  |_   _| |__   ___    ___ ___   ___ | | ___ _ __                   
| |\/| |/ _ \| '_ \| __/ _ \ |   / _` | '__| |/ _ \| |/ _ \| |_       | | | '_ \ / _ \  / __/ _ \ / _ \| |/ _ \ '__|                  
| |  | | (_) | | | | ||  __/ |__| (_| | |  | | (_) | | (_) |  _|      | | | | | |  __/ | (_| (_) | (_) | |  __/ |                     
|_|  |_|\___/|_| |_|\__\___|\____\__,_|_|  |_|\___/|_|\___/|_|       _|_|_|_| |_|\___|  \___\___/_\___/|_|\___|_|     _____     _____ 
                                                                    |  \/  | ___  _ __ | |_ ___ / ___|__ _ _ __| | __|_   _|__ |  ___|
                                                                    | |\/| |/ _ \| '_ \| __/ _ \ |   / _` | '__| |/ _ \| |/ _ \| |_   
                                                                    | |  | | (_) | | | | ||  __/ |__| (_| | |  | | (_) | | (_) |  _|  
                                                                    |_|  |_|\___/|_| |_|\__\___|\____\__,_|_|  |_|\___/|_|\___/|_|    
"""
import numpy as np
import os
import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
from ClassAdam import Adam
import AlgoToF
import scipy
import multiprocessing 
import time
import random
import sys
import math

#TODO: Note: when you wanna turn this all into a package, replace all the references back to PyToF.reference again

from color import c

def parameterise_starting_points(ToF, weights, ResultFunction):
    #note: second density entry should be nonzero
    assert(ResultFunction[1]>0), "Error: Outermost nonendpoint density is zero"
    assert(ToF.opts['N'] == len(ResultFunction)), "Error: Generated density function is not of correct length."
    #after the raw resultfunction has been generated, it needs to be preconditioned for total mass agreeing
    MassFixFactor = ToF.opts['M_init']/(-4*np.pi*scipy.integrate.simpson(ResultFunction*ToF.li**2, ToF.li))
    ResultFunction *= MassFixFactor
    if ToF.opts['verbosity'] > 1: print("MassFixFactor for initial generation was: " + str(MassFixFactor))
    #then, we need to obtain parameters from the conditioned function (note fudging factor p_alpha should be 1 now)
    params = np.zeros(ToF.opts['N']-1)
    for i in range(len(params)):
        if ResultFunction[i+1] <= ResultFunction[i]:
            params[i] = -100+math.log(weights[i]) #a number very close to zero or negative, log would be -∞
            if ToF.opts['verbosity'] > 0: print(c.WARN + 'Warning: Starting Function contained nonincreasing step. Fudged to avoid log(0) = -∞' + c.ENDC)
            continue
        params[i] = math.log(abs(weights[i]*(ResultFunction[i+1]-ResultFunction[i]))) #ensure no negatives either
    return params

def create_starting_point_fixed_jupiter(ToF, weights):
    ResultFunction = 100*np.concatenate((np.linspace(0,0.5,20),np.linspace(0.51,1,80),np.linspace(1.1,3,300),np.linspace(3.1,6,450),np.linspace(6.1,30,24),np.linspace(30.1,50,150)))
    return parameterise_starting_points(ToF, weights, ResultFunction)

def create_starting_point_fixed_earth(ToF, weights):
    ResultFunction = np.concatenate((np.linspace(0,3000,12),np.linspace(3001,4500,100),np.linspace(4501,5000,400),np.linspace(5001,10000,12),np.linspace(10001,13000,400),np.linspace(13200,13250,100)))
    return parameterise_starting_points(ToF, weights, ResultFunction)

def density(r):
    earth_radius = 6.3710e6
    radii = (1.2215e6, 3.4800e6, 5.7010e6, 5.7710e6, 5.9710e6,
            6.1510e6, 6.3466e6, 6.3560e6, 6.3680e6, earth_radius)
    densities = (
        lambda x: 13.0885 - 8.8381*x**2,
        lambda x: 12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3,
        lambda x: 7.9565 - 6.4761*x + 5.5283*x**2 - 3.0807*x**3,
        lambda x: 5.3197 - 1.4836*x,
        lambda x: 11.2494 - 8.0298*x,
        lambda x: 7.1089 - 3.8045*x,
        lambda x: 2.691 + 0.6924*x,
        2.9,
        2.6,
        1.02
    )   
    r = np.array(r)
    radius_bounds = np.concatenate(([0], radii))
    conditions = list((lower<=r) & (r<upper) for lower, upper in
                        zip(radius_bounds[:-1], radius_bounds[1:]))
    return np.piecewise(r/earth_radius, conditions, densities)

def create_starting_point_fixed_earth_better(ToF, weights):
    earth_radius = 6.3710e6
    ResultFunction = density(np.linspace(earth_radius,0,1024))
    print(ResultFunction)
    return parameterise_starting_points(ToF, weights, ResultFunction)


def subdivide(ResultFunction, lower_x, upper_x, lower_y, upper_y):
    if upper_x == lower_x:
        ResultFunction[lower_x] = random.uniform(lower_y,upper_y)
        return
    x = random.randint(lower_x ,upper_x)
    y = random.uniform(lower_y,upper_y)
    ResultFunction[x] = y
    if lower_x < x:
        subdivide(ResultFunction, lower_x, x - 1, lower_y, y)
    if x < upper_x:
        subdivide(ResultFunction, x + 1, upper_x, y, upper_y)

def create_starting_point(ToF, weights):
    N = ToF.opts['N']
    R = 6000
    ResultFunction = np.zeros(N)
    ResultFunction[-1] = R
    subdivide(ResultFunction, 1, N - 2, 0, R)
    return parameterise_starting_points(ToF, weights, ResultFunction)

@functools.cache
def multiset_nr(k, n):
    if k == 0 and n == 0: return 1
    return math.comb(k+n-1,k)

def find_permutation_nonrec(Balls2Place, NBins):
    ResultFunction = np.zeros(NBins)
    Number_of_functions = int(multiset_nr(Balls2Place, NBins))
    index = random.randint(0,Number_of_functions-1)
    Balls2Bin = 0
    Balls2OtherBins = Balls2Place
    Bin = 0
    while Bin < NBins:
        ways2place = multiset_nr(Balls2OtherBins, NBins - Bin - 1)
        if index < ways2place:
            ResultFunction[Bin] = Balls2Bin
            Balls2Bin = 0
            Bin += 1
        else:
            index -= ways2place
            Balls2Bin += 1
            Balls2OtherBins -= 1
    assert(Balls2OtherBins == 0), "Placement Error: Not all increases distributed"
    return ResultFunction

#creates a starting point chosen uniformly at random from all possible starting points, converts it into its respective parameters.
def create_starting_point_uniform_normalised(ToF, weights):
    N = ToF.opts['N']
    R = 6000
    #Resolution, ie target max density. Value chosen to be somewhat realistic TODO: make this smart
    ResultFunction = find_permutation_nonrec(R, N)
    for i in range(N-1):
        ResultFunction[i+1] = ResultFunction[i+1]+ResultFunction[i]
    return parameterise_starting_points(ToF, weights, ResultFunction)

#fixed works with the order of rhoi and li, that is, param now represents the jumps going inward
#it respects the mass and also makes the last parameter dependent on the rest to ensure equal distribution
def param_to_rho_exp_fixed(OptToF, ToF, params):
    rho = np.zeros(len(params)+1)
    for i in range(len(params)):
        rho[i+1] = rho[i]+np.exp(params[i])/OptToF.weights[i]
    MassFixFactor = ToF.opts['M_init']/(-4*np.pi*scipy.integrate.simpson(rho*ToF.li**2, ToF.li))
    rho *= MassFixFactor
    OptToF.mass_fix_factor_running_average = OptToF.update_running_average(OptToF.mass_fix_factor_running_average, MassFixFactor)
    """
    if random.random() < 0.05:
        print("MassFixFactor was: " + str(MassFixFactor))
    """
    return rho

def igradient_exp_fixed(ToF, params, gamma, weights):
    #TODO: May God and Allah stand by you
    gradient = np.zeros_like(params)
    expsum = 0 #∑_j=1 ^i   e^p_j / w_j
    for i in range(len(gradient)):
        expsum += np.exp(params[i])/weights[i]
        gradient[i] = (1 - gamma * expsum * weights[i])
    return gradient

def igradient_exp_fixed_2(ToF, params, gamma, weights):
    #TODO: May you achieve nirvana
    nto1vec = np.arange(len(params), 0, -1)
    #easy way to write sum of sum of e^pi/wi by reordering terms
    return nto1vec - gamma*np.dot(nto1vec,np.exp(params)/weights)*weights

def full_gradient(OptToF, ToF, params):
    #TODO: Not all those who wander are lost

    #Phase 1: Preliminaries
    #=============================
    time0 = time.perf_counter()

    #rhoi: now generate rhoi without mass normalisation p_α
    unnormalised_rhoi = np.zeros(len(params)+1)
    for i in range(len(params)):
        unnormalised_rhoi[i+1] = unnormalised_rhoi[i]+np.exp(params[i])/OptToF.weights[i]
    #gamma = (dl/∫)
    integral = (-scipy.integrate.simpson(unnormalised_rhoi*ToF.li**2, ToF.li))
    assert(integral > 0), "Integrated the wrong way round!"
    gamma = abs(ToF.li[2]-ToF.li[1])/(integral)
    #factor for mass cost gradient, 2*(4π∫/M-1)*4π/M*∇∫
    masscostfactor = (8*np.pi/ToF.opts['M_init'])*(4*np.pi*integral/ToF.opts['M_init']-1)
    #calculate rho
    ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, params)
    #first assert the mass is correct
    assert(abs(abs(-4*np.pi*scipy.integrate.simpson(ToF.rhoi*ToF.li**2, ToF.li)/ToF.opts['M_init'])-1)<0.01), "MassIntError: Density curve does not fit Planet Mass within 1% margin"

    #Phase 2: ToF
    #=============================
    time1 = time.perf_counter()
    #Calculate Js and SS for gradients
    call_ToF(OptToF, ToF)

    time2 = time.perf_counter()

    #Phase 3: Gradients
    #=============================
    gradient = np.zeros_like(params)

    objective_gradient = np.zeros_like(params)

    # Objective Gradient

    # This is really just in range of order. 
    # Target Js can be any length, start at J2. Our Js start at J0, thats why we can go as far as length of Target_Js but no longer than length of Js-1, because we have J0
    # We have SSs till order + 1 but again skip the first
    n = min(ToF.opts['order'] + 1 - 1, len(ToF.opts['Target_Js']))
    Flag = False
    if False and random.random() < 0.02:
        Flag = True
    gradvec = igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights)
    for i in range(n):
        temp = -(1/n)*(ToF.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*gradvec
        objective_gradient += temp
        if Flag:
            print("Magnitude of J gradient Nr " + str(2*(i+1)))
            print('{:.4e}'.format(np.linalg.norm(temp)))
            print("SSdiff:" + '{:.4e}'.format(ToF.SS[i+1][-1]-ToF.SS[i+1][-2]))
        #objective_gradient += -(1/n)*OptToF.Jratios[i]*(ToF.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*gradvec


    """
    prob = (ToF.Js[1:]-ToF.opts['Target_Js'])**2/ToF.opts['Sigma_Js']**2
    prob /= sum(prob)
    i = np.random.choice(range(n), p =  prob)
    objective_gradient += -(1/n)*(class_obj.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights)
    if random.random() < 0.02:
        print("Probabilities:")
        print(prob)
        print("Doing: " + str(i))
        print("Gradient of S_" + str(i) + 'is:')
        print((1/n)*(-1)*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights))
    """
    
    # Mass Gradient
    #∫ ~ M 
    #∇∫ = e^p_i/w_i* ∑_j=i ^n   l_j^2 == e^p_i
    #this was the point of weight correction
    mass_gradient = masscostfactor*np.exp(params)

    # Distance Gradient
    #dont go too far
    #f = 1/2||params - start_params||^2, grad = (params-start_params)ONES
    local_gradient = (params - OptToF.start_params)

    mass_gradient *= np.linalg.norm(objective_gradient)
    #local_gradient /= np.linalg.norm(local_gradient)

    #combine
    gradient = objective_gradient + OptToF.costfactor*mass_gradient + OptToF.localfactor*OptToF.localfactor*local_gradient


    if OptToF.opts['verbosity'] > 3 and random.random() < OptToF.opts['DBGshowchance']:
        print("I'm currently at:")
        print(params[::100])
        print("With rhos equal to:")
        print(ToF.rhoi[::100])
        print("Current Objective Gradient is:")
        print(objective_gradient[::100])
        print("Current Mass Corrector Gradient is:")
        print(OptToF.costfactor*mass_gradient[::100]) 
        print("Current Locality Gradient is:")
        print(OptToF.localfactor*local_gradient[::100])
        print("Current Full Gradient is:")
        print(gradient[::100])

    
    """
    if random.random()<0.03:
        print("I'm currently at:")
        print(params)
        print("With rhos equal to:")
        print(ToF.rhoi)
        print("I'm going towards:")
        print(gradient)
        print("My gamma is:")
        print(gamma)
        print("The gradient of S2 is:")
        print(igradient_exp_fixed(ToF, params, 1, gamma))
    """

    time3 = time.perf_counter()
    OptToF.timing[0] += time1-time0
    OptToF.timing[1] += time2-time1
    OptToF.timing[2] += time3-time2
    return gradient

def terrible_gradient(OptToF, ToF, params):
    #TODO: Not all those who wander are lost

    #Phase 1: Preliminaries
    #=============================
    time0 = time.perf_counter()

    #rhoi: now generate rhoi without mass normalisation p_α
    unnormalised_rhoi = np.zeros(len(params)+1)
    for i in range(len(params)):
        unnormalised_rhoi[i+1] = unnormalised_rhoi[i]+np.exp(params[i])/OptToF.weights[i]
    #gamma = (dl/∫)
    integral = (-scipy.integrate.simpson(unnormalised_rhoi*ToF.li**2, ToF.li))
    assert(integral > 0), "Integrated the wrong way round!"
    gamma = abs(ToF.li[2]-ToF.li[1])/(integral)
    #factor for mass cost gradient, 2*(4π∫/M-1)*4π/M*∇∫
    masscostfactor = (8*np.pi/ToF.opts['M_init'])*(4*np.pi*integral/ToF.opts['M_init']-1)
    #calculate rho
    ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, params)
    #first assert the mass is correct
    assert(abs(abs(-4*np.pi*scipy.integrate.simpson(ToF.rhoi*ToF.li**2, ToF.li)/ToF.opts['M_init'])-1)<0.01), "MassIntError: Density curve does not fit Planet Mass within 1% margin"

    #Phase 2: ToF
    #=============================
    time1 = time.perf_counter()
    #Calculate Js and SS for gradients
    call_ToF(OptToF, ToF)

    time2 = time.perf_counter()

    #Phase 3: Gradients
    #=============================
    gradient = np.zeros_like(params)

    objective_gradient = np.zeros_like(params)

    # Objective Gradient

    # This is really just in range of order. 
    # Target Js can be any length, start at J2. Our Js start at J0, thats why we can go as far as length of Target_Js but no longer than length of Js-1, because we have J0
    # We have SSs till order + 1 but again skip the first
    n = min(ToF.opts['order'] + 1 - 1, len(ToF.opts['Target_Js']))
    Flag = False
    if random.random() < 0.001:
        Flag = True
    SSbase = ToF.SS
    prob = (ToF.Js[1:]-ToF.opts['Target_Js'])**2/ToF.opts['Sigma_Js']**2
    prob /= sum(prob)
    nu = 1e-8
    Cost_old = calc_cost(ToF)
    for k in range(50):
        i = np.random.choice(range(n), p =  prob)
        u = (np.random.normal(0, 1, len(params)))
        jiggledparams = params + nu*u
        #calculate jiggled rhoi
        ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, jiggledparams)
        #Calculate Js and SS for gradients
        call_ToF(OptToF, ToF)
        temp = 1/10*u*(1/nu)*(calc_cost(ToF) - Cost_old)
        objective_gradient += temp
        if Flag:
            print("Probabilities:")
            print(prob)
            print("Doing: " + str(i))
            print("Gradient of S_" + str(i) + 'is:')
            print(temp)
            print("Magnitude of J gradient Nr " + str(2*(i+1)))
            print('{:.4e}'.format(np.linalg.norm(temp)))
            print("SSdiff:" + '{:.4e}'.format(ToF.SS[i+1][-1]-SSbase[i+1][-1]))
        #objective_gradient += -(1/n)*OptToF.Jratios[i]*(ToF.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*gradvec


    """
    prob = (ToF.Js[1:]-ToF.opts['Target_Js'])**2/ToF.opts['Sigma_Js']**2
    prob /= sum(prob)
    i = np.random.choice(range(n), p =  prob)
    objective_gradient += -(1/n)*(class_obj.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights)
    if random.random() < 0.02:
        print("Probabilities:")
        print(prob)
        print("Doing: " + str(i))
        print("Gradient of S_" + str(i) + 'is:')
        print((1/n)*(-1)*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights))
    """
    
    # Mass Gradient
    #∫ ~ M 
    #∇∫ = e^p_i/w_i* ∑_j=i ^n   l_j^2 == e^p_i
    #this was the point of weight correction
    mass_gradient = masscostfactor*np.exp(params)

    # Distance Gradient
    #dont go too far
    #f = 1/2||params - start_params||^2, grad = (params-start_params)ONES
    local_gradient = (params - OptToF.start_params)

    mass_gradient *= np.linalg.norm(objective_gradient)
    #local_gradient /= np.linalg.norm(local_gradient)

    #combine
    gradient = objective_gradient + OptToF.costfactor*mass_gradient + OptToF.localfactor*OptToF.localfactor*local_gradient


    if OptToF.opts['verbosity'] > 3 and random.random() < OptToF.opts['DBGshowchance']:
        print("I'm currently at:")
        print(params[::100])
        print("With rhos equal to:")
        print(ToF.rhoi[::100])
        print("Current Objective Gradient is:")
        print(objective_gradient[::100])
        print("Current Mass Corrector Gradient is:")
        print(OptToF.costfactor*mass_gradient[::100]) 
        print("Current Locality Gradient is:")
        print(OptToF.localfactor*local_gradient[::100])
        print("Current Full Gradient is:")
        print(gradient[::100])

    
    """
    if random.random()<0.03:
        print("I'm currently at:")
        print(params)
        print("With rhos equal to:")
        print(ToF.rhoi)
        print("I'm going towards:")
        print(gradient)
        print("My gamma is:")
        print(gamma)
        print("The gradient of S2 is:")
        print(igradient_exp_fixed(ToF, params, 1, gamma))
    """

    time3 = time.perf_counter()
    OptToF.timing[0] += time1-time0
    OptToF.timing[1] += time2-time1
    OptToF.timing[2] += time3-time2
    return gradient

def mcterrible_gradient(OptToF, ToF, params):
    #TODO: Not all those who wander are lost

    #Phase 1: Preliminaries
    #=============================
    time0 = time.perf_counter()

    #rhoi: now generate rhoi without mass normalisation p_α
    unnormalised_rhoi = np.zeros(len(params)+1)
    for i in range(len(params)):
        unnormalised_rhoi[i+1] = unnormalised_rhoi[i]+np.exp(params[i])/OptToF.weights[i]
    #gamma = (dl/∫)
    integral = (-scipy.integrate.simpson(unnormalised_rhoi*ToF.li**2, ToF.li))
    assert(integral > 0), "Integrated the wrong way round!"
    gamma = abs(ToF.li[2]-ToF.li[1])/(integral)
    #factor for mass cost gradient, 2*(4π∫/M-1)*4π/M*∇∫
    masscostfactor = (8*np.pi/ToF.opts['M_init'])*(4*np.pi*integral/ToF.opts['M_init']-1)
    #calculate rho
    ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, params)
    #first assert the mass is correct
    assert(abs(abs(-4*np.pi*scipy.integrate.simpson(ToF.rhoi*ToF.li**2, ToF.li)/ToF.opts['M_init'])-1)<0.01), "MassIntError: Density curve does not fit Planet Mass within 1% margin"

    #Phase 2: ToF
    #=============================
    time1 = time.perf_counter()
    #Calculate Js and SS for gradients
    call_ToF(OptToF, ToF)

    time2 = time.perf_counter()

    #Phase 3: Gradients
    #=============================
    gradient = np.zeros_like(params)

    objective_gradient = np.zeros_like(params)

    # Objective Gradient

    # This is really just in range of order. 
    # Target Js can be any length, start at J2. Our Js start at J0, thats why we can go as far as length of Target_Js but no longer than length of Js-1, because we have J0
    # We have SSs till order + 1 but again skip the first
    n = min(ToF.opts['order'] + 1 - 1, len(ToF.opts['Target_Js']))
    Flag = False
    if False and random.random() < 0.02:
        Flag = True
    SSbase = ToF.SS
    prob = (ToF.Js[1:]-ToF.opts['Target_Js'])**2/ToF.opts['Sigma_Js']**2
    prob /= sum(prob)
    i = np.random.choice(range(n), p =  prob)
    j = random.randrange(n)
    dpj = 1e-10*(np.random.random() - 0.5)
    jiggledparams = params
    jiggledparams[j] += dpj
    #calculate jiggled rhoi
    ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, jiggledparams)
    #Calculate Js and SS for gradients
    call_ToF(OptToF, ToF)
    temp = -(1/n)*OptToF.Jratios[i]*(ToF.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-SSbase[i+1][-1])/dpj
    objective_gradient[j] += temp
    if Flag:
        print("Probabilities:")
        print(prob)
        print("Doing: " + str(i))
        print("Gradient of S_" + str(i) + 'is:')
        print(temp)
        print("Magnitude of J gradient Nr " + str(2*(i+1)))
        print('{:.4e}'.format(np.linalg.norm(temp)))
        print("SSdiff:" + '{:.4e}'.format(ToF.SS[i+1][-1]-SSbase[i+1][-1]))
    #objective_gradient += -(1/n)*OptToF.Jratios[i]*(ToF.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*gradvec


    """
    prob = (ToF.Js[1:]-ToF.opts['Target_Js'])**2/ToF.opts['Sigma_Js']**2
    prob /= sum(prob)
    i = np.random.choice(range(n), p =  prob)
    objective_gradient += -(1/n)*(class_obj.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights)
    if random.random() < 0.02:
        print("Probabilities:")
        print(prob)
        print("Doing: " + str(i))
        print("Gradient of S_" + str(i) + 'is:')
        print((1/n)*(-1)*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights))
    """
    
    # Mass Gradient
    #∫ ~ M 
    #∇∫ = e^p_i/w_i* ∑_j=i ^n   l_j^2 == e^p_i
    #this was the point of weight correction
    mass_gradient = masscostfactor*np.exp(params)

    # Distance Gradient
    #dont go too far
    #f = 1/2||params - start_params||^2, grad = (params-start_params)ONES
    local_gradient = (params - OptToF.start_params)

    mass_gradient *= np.linalg.norm(objective_gradient)
    #local_gradient /= np.linalg.norm(local_gradient)

    #combine
    gradient = objective_gradient + OptToF.costfactor*mass_gradient + OptToF.localfactor*OptToF.localfactor*local_gradient


    if OptToF.opts['verbosity'] > 3 and random.random() < OptToF.opts['DBGshowchance']:
        print("I'm currently at:")
        print(params[::100])
        print("With rhos equal to:")
        print(ToF.rhoi[::100])
        print("Current Objective Gradient is:")
        print(objective_gradient[::100])
        print("Current Mass Corrector Gradient is:")
        print(OptToF.costfactor*mass_gradient[::100]) 
        print("Current Locality Gradient is:")
        print(OptToF.localfactor*local_gradient[::100])
        print("Current Full Gradient is:")
        print(gradient[::100])

    
    """
    if random.random()<0.03:
        print("I'm currently at:")
        print(params)
        print("With rhos equal to:")
        print(ToF.rhoi)
        print("I'm going towards:")
        print(gradient)
        print("My gamma is:")
        print(gamma)
        print("The gradient of S2 is:")
        print(igradient_exp_fixed(ToF, params, 1, gamma))
    """

    time3 = time.perf_counter()
    OptToF.timing[0] += time1-time0
    OptToF.timing[1] += time2-time1
    OptToF.timing[2] += time3-time2
    return gradient

def stochastic_gradient(OptToF, ToF, params):
    #TODO: Not all those who wander are lost

    #Phase 1: Preliminaries
    #=============================
    time0 = time.perf_counter()

    #rhoi: now generate rhoi without mass normalisation p_α
    unnormalised_rhoi = np.zeros(len(params)+1)
    for i in range(len(params)):
        unnormalised_rhoi[i+1] = unnormalised_rhoi[i]+np.exp(params[i])/OptToF.weights[i]
    #gamma = (dl/∫)
    integral = (-scipy.integrate.simpson(unnormalised_rhoi*ToF.li**2, ToF.li))
    assert(integral > 0), "Integrated the wrong way round!"
    gamma = abs(ToF.li[2]-ToF.li[1])/(integral)
    #factor for mass cost gradient, 2*(4π∫/M-1)*4π/M*∇∫
    masscostfactor = (8*np.pi/ToF.opts['M_init'])*(4*np.pi*integral/ToF.opts['M_init']-1)
    #calculate rho
    ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, params)
    #first assert the mass is correct
    assert(abs(abs(-4*np.pi*scipy.integrate.simpson(ToF.rhoi*ToF.li**2, ToF.li)/ToF.opts['M_init'])-1)<0.01), "MassIntError: Density curve does not fit Planet Mass within 1% margin"

    #Phase 2: ToF
    #=============================
    time1 = time.perf_counter()
    #Calculate Js and SS for gradients
    call_ToF(OptToF, ToF)

    time2 = time.perf_counter()

    #Phase 3: Gradients
    #=============================
    gradient = np.zeros_like(params)

    objective_gradient = np.zeros_like(params)

    # Objective Gradient

    # This is really just in range of order. 
    # Target Js can be any length, start at J2. Our Js start at J0, thats why we can go as far as length of Target_Js but no longer than length of Js-1, because we have J0
    # We have SSs till order + 1 but again skip the first
    n = min(ToF.opts['order'] + 1 - 1, len(ToF.opts['Target_Js']))
    prob = (ToF.Js[1:]-ToF.opts['Target_Js'])**2/ToF.opts['Sigma_Js']**2
    prob /= sum(prob)
    i = np.random.choice(range(n), p =  prob)
    objective_gradient += -(1/n)*(ToF.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights)


    """
    prob = (ToF.Js[1:]-ToF.opts['Target_Js'])**2/ToF.opts['Sigma_Js']**2
    prob /= sum(prob)
    i = np.random.choice(range(n), p =  prob)
    objective_gradient += -(1/n)*(class_obj.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights)
    if random.random() < 0.02:
        print("Probabilities:")
        print(prob)
        print("Doing: " + str(i))
        print("Gradient of S_" + str(i) + 'is:')
        print((1/n)*(-1)*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights))
    """
    
    # Mass Gradient
    #∫ ~ M 
    #∇∫ = e^p_i/w_i* ∑_j=i ^n   l_j^2 == e^p_i
    #this was the point of weight correction
    mass_gradient = masscostfactor*np.exp(params)

    # Distance Gradient
    #dont go too far
    #f = 1/2||params - start_params||^2, grad = (params-start_params)ONES
    local_gradient = (params - OptToF.start_params)

    mass_gradient *= np.linalg.norm(objective_gradient)
    #local_gradient /= np.linalg.norm(local_gradient)

    #combine
    gradient = objective_gradient + OptToF.costfactor*mass_gradient + OptToF.localfactor*OptToF.localfactor*local_gradient


    if OptToF.opts['verbosity'] > 3 and random.random() < OptToF.opts['DBGshowchance']:
        print("I'm currently at:")
        print(params[::100])
        print("With rhos equal to:")
        print(ToF.rhoi[::100])
        print("Current Objective Gradient is:")
        print(objective_gradient[::100])
        print("Current Mass Corrector Gradient is:")
        print(OptToF.costfactor*mass_gradient[::100]) 
        print("Current Locality Gradient is:")
        print(OptToF.localfactor*local_gradient[::100])
        print("Current Full Gradient is:")
        print(gradient[::100])

    
    """
    if random.random()<0.03:
        print("I'm currently at:")
        print(params)
        print("With rhos equal to:")
        print(ToF.rhoi)
        print("I'm going towards:")
        print(gradient)
        print("My gamma is:")
        print(gamma)
        print("The gradient of S2 is:")
        print(igradient_exp_fixed(ToF, params, 1, gamma))
    """

    time3 = time.perf_counter()
    OptToF.timing[0] += time1-time0
    OptToF.timing[1] += time2-time1
    OptToF.timing[2] += time3-time2
    return gradient

def giga_gradient_i_factor(ToF, n):
    #No rest for the wicked
    #calculates df2n(1)/drho*drho/dp as df/dss dss/drho drho/dp by calling df2nds which returns df/drho as an epxlicitly calculated expression of the dss/drho
    #TODO: ONLY FOURTH ORDER IMPLEMENTED
    #ds/drho is s(1)-s(prev)/rho(1)-rho(prev) = */rhoi[1]-rho[2] (rho[0] = 0)
    #numerical stability
    drho = 5*ToF.rhoi[1]-ToF.rhoi[6]-ToF.rhoi[11]-ToF.rhoi[16]-ToF.rhoi[21]-ToF.rhoi[26]
    ss_array = np.array(ToF.ss)
    diffs = np.zeros(4)
    for i in range(4):
        avg = 5*ss_array[i+1,-1]
        for k in range(5):
            avg -= ss_array[i+1,-(1+(k+1)*5)]
        diffs[i] = avg/drho
    #diffs = np.array(((ss_array[1,-1]-ss_array[1,-21])/drho, (ss_array[2,-1]-ss_array[2,-21])/drho, (ss_array[3,-1]-ss_array[3,-21])/drho, (ss_array[4,-1]-ss_array[4,-21])/drho))
    """
    dss = np.array((ToF.opts['order'])*[np.zeros(N)]).T
    dss[0] = diffs[0]
    for i in range(N-2):
        dss[i+1] = (diffs[i+1]+diffs[i])/2
    dss[N-1] = diffs[N-2]
    Dss = np.array((ToF.opts['order'])*[np.zeros((N,N))])
    for i in range(ToF.opts['order']):
        Dss[i] = np.diag(dss[:,i])
    """
    return df2nds(n, ss_array, diffs)

def df2nds(n, ss, diffs):
    s2  = ss[1,-1]; s4  = ss[2,-1]; s6  = ss[3,-1]; s8  = ss[4,-1]
    ds2  = diffs[0]; ds4  = diffs[1]; ds6  = diffs[2]; ds8  = diffs[3]
    match n:
        case 1:
            return (3*ds2)/5. + (12*2*s2*ds2)/35. + (6*3*s2**2*ds2)/175. - (184*4*s2**3*ds2)/1925. + (24*(ds2*s4+s2*ds4))/35. + (216*(2*s2*ds2*s4+s2**2*ds4))/385. + (40*2*s4*ds4)/231.

        case 2:
            return (18*2*s2*ds2)/35. + (36*3*s2**2*ds2)/77. + (486*4*s2**3*ds2)/5005. + ds4/3. + (40*(ds2*s4+s2*ds4))/77. + (6943*(2*s2*ds2*s4+s2**2*ds4))/5005. + (162*2*ds4*s4)/1001. + (90*(ds2*s6+s2*ds6))/143.

        case 3:
            return (72*3*s2**2*ds2)/143. + (432*4*s2**3*ds2)/715. + (120*(ds2*s4+s2*ds4))/143. + (216*(2*s2*ds2*s4+s2**2*ds4))/143. + (80*2*s4*ds4)/429. + (3*ds6)/13. + (336*(ds2*s6+s2*ds6))/715.

        case 4:
            return (1296*4*s2**3*ds2)/2431. + (3780*(2*s2*ds2*s4+s2**2*ds4))/2431. + (2450*2*s4*ds4)/7293. + (168*(ds2*s6+s2*ds6))/221. + (3*ds8)/17.

def giga_gradient(OptToF, ToF, params):
    #TODO: If thou gaze long into an abyss, the abyss will also gaze into thee.

    #Phase 1: Preliminaries
    #=============================
    time0 = time.perf_counter()

    #rhoi: now generate rhoi without mass normalisation p_α
    unnormalised_rhoi = np.zeros(len(params)+1)
    for i in range(len(params)):
        unnormalised_rhoi[i+1] = unnormalised_rhoi[i]+np.exp(params[i])/OptToF.weights[i]
    #gamma = (dl/∫)
    integral = (-scipy.integrate.simpson(unnormalised_rhoi*ToF.li**2, ToF.li))
    assert(integral > 0), "Integrated the wrong way round!"
    gamma = abs(ToF.li[2]-ToF.li[1])/(integral)
    #factor for mass cost gradient, 2*(4π∫/M-1)*4π/M*∇∫
    p_alpha = ToF.opts['M_init']/(4*np.pi*integral)
    masscostfactor = (8*np.pi/ToF.opts['M_init'])*(4*np.pi*integral/ToF.opts['M_init']-1)
    #calculate rho
    ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, params)
    #first assert the mass is correct
    assert(abs(abs(-4*np.pi*scipy.integrate.simpson(ToF.rhoi*ToF.li**2, ToF.li)/ToF.opts['M_init'])-1)<0.01), "MassIntError: Density curve does not fit Planet Mass within 1% margin"

    #Phase 2: ToF
    #=============================
    time1 = time.perf_counter()
    #Calculate Js and SS for gradients
    call_ToF(OptToF, ToF)

    time2 = time.perf_counter()

    #Phase 3: Gradients
    #=============================
    gradient = np.zeros_like(params)

    objective_gradient = np.zeros_like(params)

    # Objective Gradient

    # This is really just in range of order. 
    # Target Js can be any length, start at J2. Our Js start at J0, thats why we can go as far as length of Target_Js but no longer than length of Js-1, because we have J0
    # We have SSs till order + 1 but again skip the first
    n = min(ToF.opts['order'] + 1 - 1, len(ToF.opts['Target_Js']))
    Flag = False
    if random.random() < 0.02:
        Flag = True
    
    rho_bar = ToF.opts['M_init']/((4*np.pi/3)*np.max(ToF.li)**3)
    #make drho/dp matrix
    """
    N = len(params)
    drdp = np.matrix(np.zeros((N,N)))
    expvec = np.exp(params)/OptToF.weights
    sumexpvec = np.cumsum(expvec)
    for i in range(N):
        drdp[i,:] += -p_alpha*np.exp(params[i])*gamma*sumexpvec
        drdp[i,i:] += p_alpha*expvec[i:]
    """

    drdp = np.zeros_like(params)
    drdp[0] = 1
    drdp -= gamma*np.exp(params)
    drdp *= np.exp(params[0]/OptToF.weights[0])*p_alpha

    for i in range(n):
        temp = 1e6*(1/n)*OptToF.Jratios[i]*(ToF.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.rhoi[1]/rho_bar)*giga_gradient_i_factor(ToF, i+1)*drdp

        objective_gradient += temp
        if Flag:
            print("Magnitude of J gradient Nr " + str(2*(i+1)))
            print('{:.4e}'.format(np.linalg.norm(temp)))
            print("SSdiff:" + '{:.4e}'.format(ToF.SS[i+1][-1]-ToF.SS[i+1][-2]))
        #objective_gradient += -(1/n)*OptToF.Jratios[i]*(ToF.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*gradvec
    
    # Mass Gradient
    #∫ ~ M 
    #∇∫ = e^p_i/w_i* ∑_j=i ^n   l_j^2 == e^p_i
    #this was the point of weight correction
    mass_gradient = masscostfactor*np.exp(params)

    # Distance Gradient
    #dont go too far
    #f = 1/2||params - start_params||^2, grad = (params-start_params)ONES
    local_gradient = (params - OptToF.start_params)

    mass_gradient *= np.linalg.norm(objective_gradient)
    #local_gradient /= np.linalg.norm(local_gradient)

    #combine
    gradient = objective_gradient + OptToF.costfactor*mass_gradient + OptToF.localfactor*OptToF.localfactor*local_gradient

    if OptToF.opts['verbosity'] > 3 and random.random() < OptToF.opts['DBGshowchance']:
        print("I'm currently at:")
        print(params[::100])
        print("With rhos equal to:")
        print(ToF.rhoi[::100])
        print("Current Objective Gradient is:")
        print(objective_gradient[::100])
        print("Current Mass Corrector Gradient is:")
        print(OptToF.costfactor*mass_gradient[::100]) 
        print("Current Locality Gradient is:")
        print(OptToF.localfactor*local_gradient[::100])
        print("Current Full Gradient is:")
        print(gradient[::100])

    time3 = time.perf_counter()
    OptToF.timing[0] += time1-time0
    OptToF.timing[1] += time2-time1
    OptToF.timing[2] += time3-time2
    return gradient

def crackhead_gradient(OptToF, ToF, params):
    #TODO: I'm grasping at straws

    #Phase 1: Preliminaries
    #=============================
    time0 = time.perf_counter()

    #rhoi: now generate rhoi without mass normalisation p_α
    unnormalised_rhoi = np.zeros(len(params)+1)
    for i in range(len(params)):
        unnormalised_rhoi[i+1] = unnormalised_rhoi[i]+np.exp(params[i])/OptToF.weights[i]
    #gamma = (dl/∫)
    integral = (-scipy.integrate.simpson(unnormalised_rhoi*ToF.li**2, ToF.li))
    assert(integral > 0), "Integrated the wrong way round!"
    gamma = abs(ToF.li[2]-ToF.li[1])/(integral)
    p_alpha = ToF.opts['M_init']/(4*np.pi*integral)
    #factor for mass cost gradient, 2*(4π∫/M-1)*4π/M*∇∫
    masscostfactor = (8*np.pi/ToF.opts['M_init'])*(4*np.pi*integral/ToF.opts['M_init']-1)
    #calculate rho
    ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, params)
    #first assert the mass is correct
    assert(abs(abs(-4*np.pi*scipy.integrate.simpson(ToF.rhoi*ToF.li**2, ToF.li)/ToF.opts['M_init'])-1)<0.01), "MassIntError: Density curve does not fit Planet Mass within 1% margin"

    #Phase 2: ToF
    #=============================
    time1 = time.perf_counter()
    #Calculate Js and SS for gradients
    call_ToF(OptToF, ToF)

    time2 = time.perf_counter()

    #Phase 3: Gradients
    #=============================
    gradient = np.zeros_like(params)


    # This is really just in range of order. 
    # Target Js can be any length, start at J2. Our Js start at J0, thats why we can go as far as length of Target_Js but no longer than length of Js-1, because we have J0
    # We have SSs till order + 1 but again skip the first
    n = min(ToF.opts['order'] + 1 - 1, len(ToF.opts['Target_Js']))
    Flag = False
    if False and random.random() < 0.02:
        Flag = True


    objective_gradient = np.zeros_like(params)

    actual_rhoi = ToF.rhoi.copy()
    actual_li   = ToF.li.copy()
    actual_Js   = ToF.Js.copy()

    # Objective Gradient
    # We calculate the change in relation to changes of rho by simply sending drho through ToF

    ToF.li = ToF.li[1:]
    expvec = np.exp(params)/OptToF.weights
    sumexpvec = np.cumsum(expvec)
    for i in range(len(params)):
        drdpi = -p_alpha*np.exp(params[i])*gamma*sumexpvec
        drdpi[i:] += p_alpha*np.exp(params[i])/OptToF.weights[i]*np.ones(len(params)-i)
        ToF.rhoi = drdpi
        ToF.ss = None
        call_ToF(OptToF, ToF)
        for j in range(n):
            objective_gradient[i] += (1/n)*((actual_Js[j+1] - ToF.opts['Target_Js'][j])/ToF.opts['Sigma_Js'][j]**2)*ToF.Js[j+1]
            
    ToF.rhoi = actual_rhoi
    ToF.li = actual_li 
    """
    prob = (ToF.Js[1:]-ToF.opts['Target_Js'])**2/ToF.opts['Sigma_Js']**2
    prob /= sum(prob)
    i = np.random.choice(range(n), p =  prob)
    objective_gradient += -(1/n)*(class_obj.R_ratio**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights)
    if random.random() < 0.02:
        print("Probabilities:")
        print(prob)
        print("Doing: " + str(i))
        print("Gradient of S_" + str(i) + 'is:')
        print((1/n)*(-1)*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*igradient_exp_fixed_2(ToF, params, gamma, OptToF.weights))
    """
    
    # Mass Gradient
    #∫ ~ M 
    #∇∫ = e^p_i/w_i* ∑_j=i ^n   l_j^2 == e^p_i
    #this was the point of weight correction
    mass_gradient = masscostfactor*np.exp(params)

    # Distance Gradient
    #dont go too far
    #f = 1/2||params - start_params||^2, grad = (params-start_params)ONES
    local_gradient = (params - OptToF.start_params)

    mass_gradient *= np.linalg.norm(objective_gradient)
    #local_gradient /= np.linalg.norm(local_gradient)

    #combine
    gradient = objective_gradient + OptToF.costfactor*mass_gradient + OptToF.localfactor*OptToF.localfactor*local_gradient


    if OptToF.opts['verbosity'] > 3 and random.random() < OptToF.opts['DBGshowchance']:
        print("I'm currently at:")
        print(params[::100])
        print("With rhos equal to:")
        print(ToF.rhoi[::100])
        print("Current Objective Gradient is:")
        print(objective_gradient[::100])
        print("Current Mass Corrector Gradient is:")
        print(OptToF.costfactor*mass_gradient[::100]) 
        print("Current Locality Gradient is:")
        print(OptToF.localfactor*local_gradient[::100])
        print("Current Full Gradient is:")
        print(gradient[::100])

    
    """
    if random.random()<0.03:
        print("I'm currently at:")
        print(params)
        print("With rhos equal to:")
        print(ToF.rhoi)
        print("I'm going towards:")
        print(gradient)
        print("My gamma is:")
        print(gamma)
        print("The gradient of S2 is:")
        print(igradient_exp_fixed(ToF, params, 1, gamma))
    """

    time3 = time.perf_counter()
    OptToF.timing[0] += time1-time0
    OptToF.timing[1] += time2-time1
    OptToF.timing[2] += time3-time2
    return gradient

def opt_step(Optimiser, OptToF, ToF, params):
    return Optimiser.update(params, full_gradient(OptToF, ToF, params))

def terrible_step(Optimiser, OptToF, ToF, params):
    return Optimiser.update(params, terrible_gradient(OptToF, ToF, params))

def mcterrible_step(Optimiser, OptToF, ToF, params):
    return Optimiser.update(params, mcterrible_gradient(OptToF, ToF, params))

def stochastic_step(Optimiser, OptToF, ToF, params):
    return Optimiser.update(params, stochastic_gradient(OptToF, ToF, params))

def giga_step(Optimiser, OptToF, ToF, params):
    return Optimiser.update(params, giga_gradient(OptToF, ToF, params))

def crackhead_step(Optimiser, OptToF, ToF, params):
    return Optimiser.update(params, crackhead_gradient(OptToF, ToF, params))

def calc_cost(ToF):
    costJs = np.zeros_like(ToF.opts['Target_Js'])
    for i in range(min(len(ToF.opts['Target_Js']),len(ToF.Js)-1)):
        costJs[i] = (ToF.Js[i+1] - ToF.opts['Target_Js'][i])**2/ToF.opts['Sigma_Js'][i]**2
    return sum(costJs)

def call_ToF(OptToF, ToF):

    """
    Calls Algorithm from AlgoToF until either the accuray given by ToF.opts['dJ_tol'] is fulfilled 
    or ToF.opts['MaxIterHE'] is reached.
    """

    alphas = np.zeros(len(ToF.opts['alphas']))

    #Convert barotropic differential rotation parameters to Theory of Figures logic:
    if np.any(ToF.opts['alphas']):

        for i in range(len(alphas)):

            alphas[i] = 2*(i+1) * (ToF.li[0])**(2*i) * ToF.opts['alphas'][i] / ( ( ToF.m_rot_calc*ToF.opts['G']*ToF.opts['M_init'] ) / ToF.li[0]**3 ) / ToF.opts['R_init']**(2*(i+1))

    #Implement the Theory of Figures: 
    ToF.Js, out = AlgoToF.Algorithm(  ToF.li,
                                            ToF.rhoi,
                                            ToF.m_rot_calc,
                                            order       = ToF.opts['order'],
                                            nx          = ToF.opts['nx'],
                                            tol         = OptToF.ToF_convergence_tolerance,
                                            maxiter     = ToF.opts['MaxIterHE'],
                                            verbosity   = ToF.opts['verbosity'],
                                            alphas      = alphas,
                                            H           = ToF.opts['H'],
                                            ss_guesses  = ToF.ss
                                            )
    
    #Save results, flipped since AlgoToF uses a different ordering logic:
    
    ToF.A0        = out.A0
    ToF.As        = out.As #dont need these
    ToF.ss        = out.ss
    ToF.SS        = out.SS
    ToF.R_ratio   = out.R_ratio

    return



#def classify_and_save_state() TODO: Do
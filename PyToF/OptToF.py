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
from PyToF.ClassAdam import Adam
import PyToF.AlgoToF as AlgoToF
from multiprocessing import Pool 
import time


from PyToF.color import c

def _check_phys(class_obj):

    """
    This function returns True if the current solutions (class_obj.rhoi, class_obj.Pi, class_obj.Js) are
    unphysical or consisting of NaN values. Returns False if everything is ok.
    """

    #Density inversions are considered to be unphysical:
    if (np.any(np.diff(class_obj.rhoi) < 0)):

        if (class_obj.opts['verbosity'] > 1):

            print(c.WARN + 'Physicality warning: density inversion!' + c.ENDC)

        return True

    #There should be no NaN values:
    elif np.isnan(class_obj.rhoi).any() or np.isnan(class_obj.Pi).any() or np.isnan(class_obj.Js).any():

        if (class_obj.opts['verbosity'] > 1):

            print(c.WARN + 'Physicality warning: NaN values!' + c.ENDC)

        return True
    
    #There should be no negative density or pressure values:
    elif (np.min(class_obj.rhoi)<0) or (np.min(class_obj.Pi)<0):

        if (class_obj.opts['verbosity'] > 1):

            print(c.WARN + 'Physicality warning: negative density or pressure values!' + c.ENDC)

        return True

    #There should be no density or pressure values above the given maxima:
    elif (np.max(class_obj.rhoi)>class_obj.opts['rho_MAX']) or (np.max(class_obj.Pi)>class_obj.opts['P_MAX']):

        if (class_obj.opts['verbosity'] > 1):

            print(c.WARN + 'Physicality warning: density or pressure values above the given maxima!' + c.ENDC)

        return True
    
    #We passed all checks, we can return False:
    else:

        return False

def set_check_param(class_obj, fun):
    
    """
    This function allows the user to set the function that checks the parameters for the barotrope or density function.
    The function fun should have the form fun(param, give_atmosphere_index=False) and should:

    - return True if the parameters are nonsense (i.e. out of bounds)
    - return only the parameter specifiyng the location of the atmosphere if give_atmosphere_index==True
    """
    
    #Set function:
    class_obj.check_param = fun       

def dens_cost_function(class_obj, param, return_sum=True):

    """
    Calls relax_to_barotrope() and returns values specifying how far off the calculated Js are from opts['Target_Js'] given opts['Sigma_Js'].
    """

    #Infinite cost if the parameters are out of bounds:
    if class_obj.check_param(param):

        if return_sum:

            return -np.inf

        else:

            return [-np.inf]*len(class_obj.opts['Target_Js'])

    #Reset to initial conditions:
    class_obj._set_IC()

    #Get densities according to class_obj.density_function():
    class_obj.rhoi = class_obj.density_function(class_obj.li, param=param)

    #Update the internal parameters:
    class_obj.dens_param_calc = param

    #Call relax_to_density():
    try:

        it = class_obj.relax_to_density(fixradius=True, fixmass=True, fixrot=True, pressurize=True)
    
    except:

        if return_sum:

            return -np.inf

        else:

            return [-np.inf]*len(class_obj.opts['Target_Js'])

    #Infinite cost if the result is unphysical:
    if _check_phys(class_obj) or it == class_obj.opts['MaxIterDen']:

        if return_sum:

            return -np.inf

        else:

            return [-np.inf]*len(class_obj.opts['Target_Js'])

    #Calculate the cost, i.e. the deviation from the target values:
    costJs = np.zeros_like(class_obj.opts['Target_Js'])

    for i in range(len(costJs)):

        costJs[i] = -(class_obj.Js[i+1] - class_obj.opts['Target_Js'][i])**2/class_obj.opts['Sigma_Js'][i]**2

    #Return the cost value:
    if return_sum:

        return np.sum(costJs)

    else:

        return costJs

def create_starting_points(class_obj, nwalkers):
    return np.random.rand(len(class_obj.rhoi)-1)

#TODO: THIS CALCULATION IS WRONG (no renormalisation)
def param_to_rho(params):
    rho = np.zeros(len(params)+1)
    for i in range(len(params)):
        rho[i+1] = rho[i]+max(0,params[i])
    return rho

def param_to_rho_exp(params):
    rho = np.zeros(len(params)+1)
    for i in range(len(params)):
        rho[i+1] = rho[i]+np.exp(params[i])
    return rho

#TODO: THIS GRADIENT IS EVIL
def igradient(class_obj, params, S_i):
    gradient = np.zeros_like(params)
    inv = 1/params
    sum = 0
    n = len(gradient)
    for i in range(n):
        sum += inv[n-1-i]
        gradient[n-1-i] = sum
    gradient *= (class_obj.SS[S_i][-1]-class_obj.SS[S_i][-2])
    return gradient

def igradient_exp(class_obj, params, S_i, mothermatrix):
    #TODO: God help you
    gradient = mothermatrix.dot(params)
    gradient *= (class_obj.SS[S_i][-1]-class_obj.SS[S_i][-2])
    gradient /= np.sum(np.exp(params))**2
    return gradient

def full_gradient(class_obj, params, mothermatrix):
    class_obj.rhoi = param_to_rho_exp(params)
    call_ToF(class_obj)
    gradient = np.zeros_like(params)
    #TODO: This is stupid and horrible
    n = min(len(class_obj.Js)-1, len(class_obj.opts['Target_Js']), class_obj.opts['order']+1, len(class_obj.SS))
    for i in range(n):
        gradient += -(class_obj.R_ratio**(-2*i))*(class_obj.Js[i+1] - class_obj.opts['Target_Js'][i])*igradient_exp(class_obj, params, i, mothermatrix)
    
    gradient /= n
    return gradient

def calc_cost(class_obj):
    costJs = np.zeros_like(class_obj.opts['Target_Js'])
    for i in range(len(costJs)):
        costJs[i] = (class_obj.Js[i+1] - class_obj.opts['Target_Js'][i])**2/class_obj.opts['Sigma_Js'][i]**2
    return sum(costJs)

def call_ToF(class_obj):

    """
    Calls Algorithm from AlgoToF until either the accuray given by class_obj.opts['dJ_tol'] is fulfilled 
    or class_obj.opts['MaxIterHE'] is reached.
    """

    alphas = np.zeros(len(class_obj.opts['alphas']))

    #Convert barotropic differential rotation parameters to Theory of Figures logic:
    if np.any(class_obj.opts['alphas']):

        for i in range(len(alphas)):

            alphas[i] = 2*(i+1) * (class_obj.li[0])**(2*i) * class_obj.opts['alphas'][i] / ( ( class_obj.m_rot_calc*class_obj.opts['G']*class_obj.opts['M_init'] ) / class_obj.li[0]**3 ) / class_obj.opts['R_init']**(2*(i+1))

    tic = time.time()

    #Implement the Theory of Figures: 
    class_obj.Js, out = AlgoToF.Algorithm(  class_obj.li,
                                            class_obj.rhoi,
                                            class_obj.m_rot_calc,
                                            order       = class_obj.opts['order'],
                                            nx          = class_obj.opts['nx'],
                                            tol         = class_obj.opts['dJ_tol'],
                                            maxiter     = class_obj.opts['MaxIterHE'],
                                            verbosity   = class_obj.opts['verbosity'],
                                            alphas      = alphas,
                                            H           = class_obj.opts['H'],
                                            ss_guesses  = class_obj.ss
                                            )
    
    toc = time.time()
    """
    #Verbosity output:
    if (class_obj.opts['verbosity'] > 2):
        
        print()
        print(c.INFO + 'ToF calculation done in ' + c.NUMB + '{:.2e}'.format(toc-tic) + c.INFO + ' seconds.' + c.ENDC)
    """
    #Save results, flipped since AlgoToF uses a different ordering logic: TODO: WHY?
    
    class_obj.A0        = out.A0
    class_obj.As        = out.As
    class_obj.ss        = out.ss
    class_obj.SS        = out.SS
    class_obj.R_ratio   = out.R_ratio

    return

def run_dens_opt(class_obj, nwalkers, steps, Ncores=8, parallelize=False):

    """
    Uses the Adam algorithm for the given amount of steps and creates nwalkers many parameter candidates
    (i.e. walkers) that will be optimised over the allowed parameter space (constrained by check_phys() and check_param()).
    """

    #Find a random set of starting parameters:
    param_0 = create_starting_points(class_obj, nwalkers)
    #standard start
    #param_0 = np.ones(len(class_obj.rhoi)-1)/len(class_obj.rhoi)


    #Set up iterative procedure:
    i = 0
    #TODO: Make multiple
    """
    while i < nwalkers:

        #Propose a new candidate:
        param_0[i,:]        = np.random.rand(1, len(class_obj.opts['dens_param_init'])) * class_obj.opts['dens_param_init']

        #Propose new candidates as long the previous one is out of bounds:
        while class_obj.check_param(param_0[i,:]):

            param_0[i,:]    = np.random.rand(1, len(class_obj.opts['dens_param_init'])) * class_obj.opts['dens_param_init']
        
        #Reset to initial conditions:
        class_obj._set_IC()

        #Get densities according to class_obj.density_function():
        class_obj.rhoi = class_obj.density_function(class_obj.li, param=param_0[i,:])

        #Update the internal parameters:
        class_obj.dens_param_calc = param_0[i,:]

        #Call relax_to_density():
        try:

            it = class_obj.relax_to_density(fixradius=True, fixmass=True, fixrot=True, pressurize=True)

        except:

            continue

        #Only proceed if the candidate also yields a physical solution:
        if (not _check_phys(class_obj)) and it < class_obj.opts['MaxIterDen']:

            if (class_obj.opts['verbosity'] > 0):

                print(c.INFO + 'Generated initial conditions for walker number: ' + c.NUMB + str(i) + c.ENDC)

            i = i+1
    """
    #Do the optimisation algorithm in a parallel manner on multiple cores (TODO: UNSUPPORTED):

    if parallelize:

        with Pool(processes = Ncores) as pool:

            sampler  = emcee.EnsembleSampler(nwalkers, len(class_obj.opts['dens_param_init']), dens_log_prob, args=[class_obj], pool=pool)

            #Run the Markov Chain Monte Carlo class from emcee:
            index    = 0; autocorr = np.empty(steps); old_tau  = np.inf

            for sample in sampler.sample(param_0, iterations=steps, progress=progress):

                if sampler.iteration % 100:
                    continue

                tau             = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index           = index + 1

                converged       = np.all(tau*100 < sampler.iteration) and np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau
            
            state = sample


    #Do the optimisation algorithm on a single core:
    else:
        #TODO: Expose options to user
        #Set up the Adam Optimiser class:
        AdamOptimiser = Adam(learning_rate=0.30758234) #This learning rate has been revealed to me by god

        params = param_0

        #mothermatrix
        n = len(params)
        ascending_vector = np.arange(n)
        ones_vector = np.ones(n)
        mothermatrix = np.zeros((n,n))
        for i in range(n):
            mothermatrix[i] = ascending_vector - i*ones_vector


        cost_vector = [calc_cost(class_obj)]

        if (class_obj.opts['verbosity'] > 0):

            print(c.INFO + 'Running optimisation: ' + c.ENDC)

        #divide steps into epochs
        epochs = 10
        steps //= epochs

        plt.subplot(2, 2, 1)
        plt.plot(param_to_rho_exp(params))

        for epoch in range(epochs):
            print(c.INFO + 'Starting epoch number: ' + c.NUMB + str(epoch+1) + '/' + str(epochs) + c.ENDC)
            print(c.INFO + 'Current cost: ' + c.NUMB + '{:.2e}'.format(calc_cost(class_obj)) + c.ENDC)

            for step in tqdm(range(steps), position=0, leave=True):
                params
                params = AdamOptimiser.update(params, full_gradient(class_obj, params, mothermatrix))
                cost_vector.append(calc_cost(class_obj))

        plt.subplot(2, 2, 2)
        plt.plot(param_to_rho_exp(params))

        state = params


    if (class_obj.opts['verbosity'] > 0):
        
        print(c.INFO + 'Final cost: ' + c.NUMB + '{:.2e}'.format(calc_cost(class_obj)) + c.ENDC)

        #print(cost_vector)
        plt.subplot(2, 2, 3)
        plt.semilogy(np.array(cost_vector))
        plt.show()


    return state[0]

def classify_and_save_state(class_obj, state, what_model, what_save='none', log_CGS_units=False, path_name=os.getcwd(), file_name='walker'):

    """
    Classifies the nwalkers many parameter candidates according to their physicality, i.e. by checking if they agree with the physical data within 1 sigma.
    Also allows for saving .txt files with the radii, densities and pressures within the planet generated by the walker parameters.
    """

    #Storage arrays: 
    matches_observed_Js = []
    rho_maxs            = []

    #Loop through walkers:
    for i in range(len(state[:,0])):

        #Calculate the physical cost:
        if what_model == 'baro':

            costJs = baro_cost_function(class_obj, state[i,:], return_sum=False)

        elif what_model == 'dens':

            costJs = dens_cost_function(class_obj, state[i,:], return_sum=False)

        else:

            raise KeyError(c.WARN + 'Invalid keyword for what_model! Use \'baro\' or \'dens\'.' + c.ENDC)

        #Check if there is agreement with physical data within 1 sigma:
        if (np.abs(costJs)<1e0).all():

            output_color = c.GOOD
            matches_observed_Js.append(True)
            
        else:

            output_color = c.WARN
            matches_observed_Js.append(False)

        #Save .txt file if wanted by the user:
        if what_save=='all':

            if log_CGS_units:

                np.savetxt(path_name + '/' + file_name + '_CGS_' + str(i) + '.txt', np.transpose(np.array([ class_obj.li * 100, 
                                                                                                            np.log10(class_obj.rhoi) - 3, 
                                                                                                            np.log10(class_obj.Pi) + 1])))

            else:

                np.savetxt(path_name + '/' + file_name + '_SI_' + str(i) + '.txt',  np.transpose(np.array([ class_obj.li, 
                                                                                                            class_obj.rhoi, 
                                                                                                            class_obj.Pi])))
        
        elif what_save=='good' and (np.abs(costJs)<1e0).all(): 

            if log_CGS_units:

                np.savetxt(path_name + '/' + file_name + '_CGS_' + str(i) + '.txt', np.transpose(np.array([ class_obj.li * 100, 
                                                                                                            np.log10(class_obj.rhoi) - 3, 
                                                                                                            np.log10(class_obj.Pi) + 1])))

            else:

                np.savetxt(path_name + '/' + file_name + '_SI_' + str(i) + '.txt',  np.transpose(np.array([ class_obj.li, 
                                                                                                            class_obj.rhoi, 
                                                                                                            class_obj.Pi])))

        elif what_save=='none':

            pass
        
        else:

            raise KeyError(c.WARN + 'Invalid keyword for what_save! Use \'all\' or \'good\' or \'none\'.' + c.ENDC)

        #Update storage arrays:
        rho_maxs.append(np.max(class_obj.rhoi))

        #Verbositiy output:
        if class_obj.opts['verbosity'] > 0:

            print()
            print(output_color + 'Walker #' + c.NUMB  + str(i) + output_color + ' yields a total cost of ' + c.NUMB + '{:.2e}'.format(np.sum(costJs)) + output_color + '.' + c.ENDC)
            string1 = '         '
            string2 = ''
            string3 = ''
            for j in range(len(costJs)):
                string1 += c.get(np.abs(costJs[j])<1e0) + '     J'+str(2*(j+1))+'     '
                string2 += '{:.5e}'.format(class_obj.Js[j+1]) + ' '
                string3 += '{:.5e}'.format(class_obj.opts['Target_Js'][j]) + ' '
            print(c.INFO + string1 + c.ENDC)
            print(c.INFO + 'My code: ' + c.NUMB + string2 + c.ENDC)
            print(c.INFO + 'Target:  ' + c.NUMB + string3 + c.ENDC)
            print()

    return np.array(matches_observed_Js), np.array(rho_maxs)
 
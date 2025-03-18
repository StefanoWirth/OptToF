###########################################################
# Author of this version: Stefano Wirth - stwirth@ethz.ch #
###########################################################

import numpy as np
import matplotlib.pyplot as plt
from ClassAdam import Adam
from OptToF import *
from StartGen import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import random
from os import cpu_count, getpid
from colorhash import ColorHash

from color import c

class OptToF:

    """
    This class contains the optimisation routine to run a descent method on planet data using the Theory of Figures.
    Note: Only optimisation concerns are wrapped here, for ToF one should configure their instance of ClassToF
    """

    def _default_opts(self, kwargs):

        """
        This function implements the standard options used for the OptToF class,
        except for the kwargs given by the user.
        """

        opts      =    {'verbosity':                    1,      #Higher numbers lead to more verbosity output in the console
                        'steps':                        1000,
                        'epoch size':                   50,
                        'cores':                        cpu_count() - 1, #TODO
                        'parallelize':                  True,
                        'time':                         True,
                        'figures':                      True,
                        'learning rate':                0.01,    
                        'costfactor':                   1e0,   
                        'localfactor':                  0,
                        'rolling average forgetfulness':0.7,
                        'early stopping':               True,
                        'convergence limit':            0.0005,
                        'ToF convergence tolerance':    1e-10,
                        'DBGshowchance':                0,              
                        'kitty':                        0.001
                        }

        #Update the standard numerical parameters with the user input provided via the kwargs
        for kw, v in kwargs.items():

            if kw in opts:

                opts[kw] = v

            else:

                print(str(kw) + c.WARN + ' is an invalid keyword!' + c.ENDC)

        return opts

    def _set_IC(self):

        """
        Initializes...
        TODO: write 
                #vector storing time taken for each subtask in seconds
        - dens_param_calc:  Possibly updated parameters used by the density function set via set_density_function()
        """

        self.improvement_running_average    = 1
        self.mass_fix_factor_running_average= 1
        self.timing                         = [0, 0, 0, 0, 0]         # Rho | Integral | Factors | ToF | Gradients
        self.convergence_strikes            = 0
        self.runtime_running_average        = 10
        self.learning_rate                  = self.opts['learning rate']
        self.costfactor                     = self.opts['costfactor']
        self.localfactor                    = self.opts['localfactor']
        self.start_params                   = None



    def __init__(self, **kwargs):

        """
        Initializes the OptToF class. Options can be provided via **kwargs, otherwise the default options from _default_opts() will be implemented.
        """

        #Set initial values:
        self.opts  = self._default_opts(kwargs)
        self._set_IC()

    def run(self, ToF):

        """
        Uses the Adam algorithm for the given amount of steps on the given number of cores.
        Each process will optimise until stopped.
        """

        #Set weights for rhoi
        self.weights = np.zeros(ToF.opts['N']-1)
        li2 = ToF.li**2 #∑_j=i ^n   l_j^2
        for i in range(len(self.weights)):
            self.weights[:i+1] += li2[i+1]*np.ones(i+1)

        self.ToF_convergence_tolerance      = min(self.opts['ToF convergence tolerance'], 0.1*min(ToF.opts['Sigma_Js']))

        #Do the optimisation algorithm in a parallel manner on multiple cores:
        if self.opts['parallelize']:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_opt, self, ToF) for core in range(self.opts['cores'])]
                for future in as_completed(futures):
                    future.result()

        #Do the optimisation algorithm on a single core:
        else:
            run_opt(self, ToF)

    def update_running_average(self, average, new):
        return self.opts['rolling average forgetfulness']*average + (1-self.opts['rolling average forgetfulness'])*new

    def huh(self):
        print()
        print(c.INFO + 'A confused little kitten has stumbled onto the terminal!' + c.ENDC)
        print()
        print()
        print('                                                     ／l、             ')
        print('                                                   （ﾟ､ ｡ ７         ')
        print('                                                     |   ~ヽ       ')
        print('                                                     じしf_,)ノ')
        print('\033[92m'+'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww' + c.ENDC)
        print()

def run_opt(OptToF, ToF):

    #random.seed(3)
    np.set_printoptions(formatter={'float_kind':'{:.5e}'.format})

    #Find a random set of starting parameters:
    tic = time.perf_counter()
    OptToF.start_params = create_starting_point(ToF, OptToF.weights)
    toc = time.perf_counter()

    if (OptToF.opts['verbosity'] > 2) and OptToF.opts['time']:
        print(c.INFO + 'Starting distribution generated in ' + c.NUMB + '{:.6f}'.format(toc-tic) + c.INFO + ' seconds.' +c.ENDC)

    params = OptToF.start_params.copy()

    #Set up the Adam Optimiser class
    AdamOptimiser = Adam(learning_rate=OptToF.learning_rate) #This learning rate has been revealed to me by god

    if (OptToF.opts['verbosity'] > 0) and OptToF.opts['parallelize'] == False:
        print()
        print(c.INFO + '==============================================================================' + c.ENDC)
        print(c.INFO + '                           Beginning optimisation                           ' + c.ENDC)
        print(c.INFO + '==============================================================================' + c.ENDC)
        print()
    
    if (OptToF.opts['verbosity'] > 0) and OptToF.opts['parallelize'] == True:
        r, g, b = ColorHash(getpid()).rgb
        cPID = '\033[38;2;' + str(r) + ';' + str(g) + ';' + str(b) + 'm'
        print()
        print(c.INFO + '==============================================================================' + c.ENDC)
        print(c.INFO + '              Beginning optimisation for process ID ' + cPID + str(getpid()) +  c.ENDC)
        print(c.INFO + '==============================================================================' + c.ENDC)
        print()

    #divide steps into epochs of epoch size each
    epochs = OptToF.opts['steps'] // OptToF.opts['epoch size']
    steps = OptToF.opts['epoch size']
    #setup plots
    figure, (devax, perax) = plt.subplots(1, 2)
    devax.plot(param_to_rho_exp_fixed(OptToF, ToF, params), color = 'red')
    #calculate first cost
    ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, params)
    call_ToF(OptToF, ToF)
    new_cost = calc_cost(ToF)
    cost_vector = [new_cost]

    total_start_time = time.perf_counter()

    for epoch in range(epochs):

        old_cost = new_cost
        new_cost = calc_cost(ToF)
        
        if (epoch + 1) % 3 == 0:
            devax.plot(param_to_rho_exp_fixed(OptToF, ToF, params), color = 'b', alpha = (epoch + 1)/epochs)
        
        #Verbosity
        if (OptToF.opts['verbosity'] > 0):
            print()
            r, g, b = ColorHash(getpid()).rgb
            cPID = '\033[38;2;' + str(r) + ';' + str(g) + ';' + str(b) + 'm'
            if OptToF.opts['parallelize'] == True: print(c.INFO + '                             Process ID: ' + cPID + str(getpid()) + c.ENDC)
            print(c.INFO + '                  Starting epoch number: ' + c.NUMB + str(epoch+1) + '/' + str(epochs) + c.ENDC)
        if (OptToF.opts['verbosity'] > 1):
            print(c.INFO + '                           Current cost: ' + c.NUMB + '{:.2e}'.format(new_cost) + c.INFO + '  A reduction of: ' + c.NUMB + '{:.2%}'.format(old_cost/new_cost-1) + c.ENDC)
            if epoch > 0 and OptToF.opts['time']:
                OptToF.runtime_running_average = OptToF.update_running_average(OptToF.runtime_running_average, (toc-tic))
                print(c.INFO + '                        Last epoch took: ' + c.NUMB + '{:.2f}'.format(toc-tic) + 's' + c.INFO + '     Time remaining: ' + c.NUMB + '{:.2f}'.format(OptToF.runtime_running_average*(epochs-epoch)) + 's' +c.ENDC)
        if OptToF.opts['verbosity'] > 2:
            print(c.INFO + '                  Current learning rate: ' + c.NUMB + str(AdamOptimiser.learning_rate) + c.ENDC)
            print(c.INFO + '               Current mass cost factor: ' + c.NUMB + '{:.0e}'.format((OptToF.costfactor)) + c.ENDC)
            print(c.INFO + '        Mass fix factor running average: ' + c.NUMB + '{:.6f}'.format((OptToF.mass_fix_factor_running_average)) + c.ENDC)
            print(c.INFO + '            Improvement running average: ' + c.NUMB + '{:.6f}'.format((OptToF.improvement_running_average)) + c.ENDC)
        if OptToF.opts['verbosity'] > 1:
            print(c.INFO + '                 J explanation strength:'
                + c.INFO + ' J2: ' + c.NUMB + '{:.3f}'.format((ToF.Js[1]-ToF.opts['Target_Js'][0])**2/ToF.opts['Sigma_Js'][0]**2)
                + c.INFO + ' J4: ' + c.NUMB + '{:.3f}'.format((ToF.Js[2]-ToF.opts['Target_Js'][1])**2/ToF.opts['Sigma_Js'][1]**2)
                + c.INFO + ' J6: ' + c.NUMB + '{:.3f}'.format((ToF.Js[3]-ToF.opts['Target_Js'][2])**2/ToF.opts['Sigma_Js'][2]**2)
                + c.INFO + ' J8: ' + c.NUMB + '{:.3f}'.format((ToF.Js[4]-ToF.opts['Target_Js'][3])**2/ToF.opts['Sigma_Js'][3]**2)
                + c.ENDC)
        if abs(new_cost/old_cost-1) < OptToF.opts['convergence limit']: OptToF.convergence_strikes += 1
        else: OptToF.convergence_strikes = 0
        if OptToF.opts['early stopping'] and OptToF.convergence_strikes >= 3:
                print(c.INFO + 'Convergence detected. Terminating.' + c.ENDC)
                break
        if random.random() < OptToF.opts['kitty'] and OptToF.opts['verbosity'] > 0:

            OptToF.huh()

        tic = time.perf_counter()

        #====================================== MAIN OPTIMISATION LOOP ======================================

        for step in range(steps):

            params = opt_step(AdamOptimiser, OptToF, ToF, params)
            params = fix_params(params, OptToF.weights, 0.5)
            cost_vector.append(calc_cost(ToF))
            OptToF.improvement_running_average = OptToF.update_running_average(OptToF.improvement_running_average, cost_vector[-1]/cost_vector[-2])

        #====================================== MAIN OPTIMISATION LOOP ======================================

        toc = time.perf_counter()

    total_stop_time = time.perf_counter()

    if (OptToF.opts['verbosity'] > 0):
        if OptToF.opts['parallelize'] == False:
            print()
            print(c.INFO + '==============================================================================' + c.ENDC)
            print(c.INFO + '                            Optimisation ended                                ' + c.ENDC)
            print(c.INFO + '==============================================================================' + c.ENDC)
            print()
        
        if OptToF.opts['parallelize'] == True:
            r, g, b = ColorHash(getpid()).rgb
            cPID = '\033[38;2;' + str(r) + ';' + str(g) + ';' + str(b) + 'm'
            print()
            print(c.INFO + '==============================================================================' + c.ENDC)
            print(c.INFO + '                  Optimisation ended for process ID ' + cPID + str(getpid()) +  c.ENDC)
            print(c.INFO + '==============================================================================' + c.ENDC)
            print()

        print(c.INFO + 'Final cost: ' + c.NUMB + '{:.2e}'.format(calc_cost(ToF)) + c.ENDC)
        print()

        print(c.INFO + 'Final Js:   ' + c.NUMB, end = ' ')
        print(ToF.Js[1:])
        print(c.INFO + 'Target Js:  ' + c.NUMB, end = ' ')
        print(ToF.opts['Target_Js'])
        print(c.INFO + 'Difference: ' + c.NUMB, end = ' ')
        print(ToF.Js[1:]-ToF.opts['Target_Js'])
        print(c.INFO + 'Tolerance:  ' + c.NUMB, end = ' ')
        print(ToF.opts['Sigma_Js'])

        print(c.ENDC)
        # Rho | Integral | Factors | ToF | Gradients
        if OptToF.opts['time']:
            print(c.INFO + 'Total time:                         ' + c.NUMB + '{:.2f}'.format(total_stop_time-total_start_time) + c.INFO + ' seconds.' +c.ENDC)
            print(c.INFO + 'Time for rho calculation:           ' + c.NUMB + '{:.2f}'.format(OptToF.timing[0]) + c.INFO + ' seconds.' +c.ENDC)
            print(c.INFO + 'Time for integral calculations:     ' + c.NUMB + '{:.2f}'.format(OptToF.timing[1]) + c.INFO + ' seconds.' +c.ENDC)
            print(c.INFO + 'Time for factor calculations:       ' + c.NUMB + '{:.2f}'.format(OptToF.timing[2]) + c.INFO + ' seconds.' +c.ENDC)
            print(c.INFO + 'Time for ToF calculations:          ' + c.NUMB + '{:.2f}'.format(OptToF.timing[3]) + c.INFO + ' seconds.' +c.ENDC)
            print(c.INFO + 'Time for gradient calculations:     ' + c.NUMB + '{:.2f}'.format(OptToF.timing[4]) + c.INFO + ' seconds.' +c.ENDC)

        perax.semilogy(np.array(cost_vector))
        if OptToF.opts['figures']: plt.show()

    return
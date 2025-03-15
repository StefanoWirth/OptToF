###########################################################
# Author of this version: Stefano Wirth - stwirth@ethz.ch #
###########################################################

import numpy as np
import os
import functools
from multiprocessing import Pool 
import matplotlib.pyplot as plt
from ClassAdam import Adam
from OptToF import *
from StartGen import *
import multiprocessing 
import time
import random
import sys
from torch import optim

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
                        'cores':                        multiprocessing.cpu_count()-1,
                        'parallelize':                  True,
                        'time':                         True,
                        'learning rate':                0.1,    #actually just direct learning rate
                        'costfactor':                   1e18,   #actually just direct costfactor
                        'localfactor':                  0,
                        'improvement strictness limit': 0.01,
                        'improvement leniency limit':   0.1,
                        'mass strictness limit':        2,
                        'mass leniency limit':          10,
                        'rolling average forgetfulness':0.7,
                        'early stopping':               False,
                        'convergence limit':            0.0005,
                        'optimum reached limit':        1,
                        'improve order':                True,
                        'improve precision':            True,
                        'ToF convergence tolerance':    1e-8,
                        'precision increase cost threshold':1e3, #if cost reduces below this, we increase the sigma precision by one magnitude
                        'precision start fudge factor': 1e4,     #must be e^n
                        'DBGshowchance':                0.02,              
                        'kitty':                        0.005
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

        self.timing                         = [0, 0, 0]         # everything before calling ToF | ToF | everything after calling tof

        self.cost_vector                    = []

        self.convergence_strikes            = 0

        self.runtime_running_average        = 10
        
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

        self.ToF_convergence_tolerance      = max(self.opts['ToF convergence tolerance'], 0.1*min(ToF.opts['Sigma_Js']))

        precision_fudge_factor              = self.opts['precision start fudge factor']

        if self.opts['improve precision']:  ToF.opts['Sigma_Js'] *= precision_fudge_factor

        self.learning_rate                  = self.opts['learning rate']

        self.costfactor                     = self.opts['costfactor']

        self.localfactor                    = self.opts['localfactor']

        #random.seed(5)

        #weights for rhoi
        self.weights = np.zeros(ToF.opts['N']-1)
        li2 = ToF.li**2 #∑_j=i ^n   l_j^2
        for i in range(len(self.weights)):
            self.weights[:i+1] += li2[i+1]*np.ones(i+1)
        #Find a random set of starting parameters:
        tic = time.perf_counter()
        self.start_params = create_starting_point(ToF, self.weights)
        toc = time.perf_counter()

        if (self.opts['verbosity'] > 1) and self.opts['time']:
            print(c.INFO + 'Starting distribution generated in ' + c.NUMB + '{:.6f}'.format(toc-tic) + c.INFO + ' seconds.' +c.ENDC)

        params = self.start_params.copy()


        #standard start
        #param_0 = np.ones(len(ToF.rhoi)-1)/len(ToF.rhoi)

        #Do the optimisation algorithm in a parallel manner on multiple cores (TODO: UNSUPPORTED):
        if self.opts['parallelize']:
            with multiprocessing.Pool(processes = self.opts['cores']) as pool:
                #TODO: Expose options to user
                #Set up the Adam Optimiser class:
                AdamOptimiser = Adam(learning_rate=self.learning_rate) #This learning rate has been revealed to me by god
                print(multiprocessing.current_process().name)

        #Do the optimisation algorithm on a single core:
        else:
            #TODO: Expose options to user
            #Set up the Adam Optimiser class: with decay this was learning_rate=0.30758234

            AdamOptimiser = Adam(learning_rate=self.learning_rate) #This learning rate has been revealed to me by god

            #model = ToFModel(params)
            #opt = optim.Adam(model.parameters(), lr=self.learning_rate, differentiable=True)

            self.cost_vector = [calc_cost(ToF)]

            if (self.opts['verbosity'] > 0):
                
                print(c.INFO + '==============================================================================' + c.ENDC)
                print(c.INFO + '                           Beginning optimisation                           ' + c.ENDC)
                print(c.INFO + '==============================================================================' + c.ENDC)
                print()

            #divide steps into epochs of 50 each
            epochs = self.opts['steps'] // self.opts['epoch size']
            steps = self.opts['epoch size']
            develop = plt.figure()
            hate = develop.add_subplot()
            final, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            ax1.plot(param_to_rho_exp_fixed(self, ToF, params))
            hate.plot(param_to_rho_exp_fixed(self, ToF, params), color = 'red')
            new_cost = calc_cost(ToF)

            total_start_time = time.perf_counter()

            done = False

            for epoch in range(epochs):
                old_cost = new_cost
                new_cost = calc_cost(ToF)
                
                if (epoch + 1) % 3 == 0:
                    hate.plot(param_to_rho_exp_fixed(self, ToF, params), color = 'b', alpha = (epoch + 1)/epochs)
                
                if (self.opts['verbosity'] > 0):
                    print('\n')
                    print(c.INFO + '                  Starting epoch number: ' + c.NUMB + str(epoch+1) + '/' + str(epochs) + c.ENDC)

                if (self.opts['verbosity'] > 1):
                    
                    print(c.INFO + '                           Current cost: ' + c.NUMB + '{:.2e}'.format(new_cost) + c.INFO + '  A reduction of: ' + c.NUMB + '{:.2%}'.format(old_cost/new_cost-1) + c.ENDC)
                    #print(c.INFO + '        Cost reduction since last epoch: ' + c.NUMB + '{:.2%}'.format(old_cost/new_cost-1) + c.ENDC)
                    if epoch > 0 and self.opts['time']:
                        self.runtime_running_average = self.update_running_average(self.runtime_running_average, (toc-tic))
                        print(c.INFO + '                        Last epoch took: ' + c.NUMB + '{:.2f}'.format(toc-tic) + 's' + c.INFO + '     Time remaining: ' + c.NUMB + '{:.2f}'.format(self.runtime_running_average*(epochs-epoch)) + 's' +c.ENDC)
                        #print(c.INFO + '               Estimated time remaining: ' + c.NUMB + '{:.2f}'.format(self.runtime_running_average*(epochs-epoch)) + c.INFO + ' seconds.' +c.ENDC)

                if self.opts['verbosity'] > 2:

                    print(c.INFO + '                  Current learning rate: ' + c.NUMB + str(AdamOptimiser.learning_rate) + c.ENDC)
                    print(c.INFO + '               Current mass cost factor: ' + c.NUMB + '{:.0e}'.format((self.costfactor)) + c.ENDC)
                    print(c.INFO + '        Mass fix factor running average: ' + c.NUMB + '{:.6f}'.format((self.mass_fix_factor_running_average)) + c.ENDC)
                    print(c.INFO + '            Improvement running average: ' + c.NUMB + '{:.6f}'.format((self.improvement_running_average)) + c.ENDC)
                    if self.opts['improve precision']: print(c.INFO + '         Current precision fudge factor: ' + c.NUMB + str(precision_fudge_factor) + c.ENDC)

                if self.opts['verbosity'] > 3:
                    print(c.INFO + '                 J explanation strength: '
                        + c.INFO + ' J2: ' + c.NUMB + '{:.2f}'.format((ToF.Js[1]-ToF.opts['Target_Js'][0])**2/ToF.opts['Sigma_Js'][0]**2)
                        + c.INFO + ' J4: ' + c.NUMB + '{:.2f}'.format((ToF.Js[2]-ToF.opts['Target_Js'][1])**2/ToF.opts['Sigma_Js'][1]**2)
                        + c.INFO + ' J6: ' + c.NUMB + '{:.2f}'.format((ToF.Js[3]-ToF.opts['Target_Js'][2])**2/ToF.opts['Sigma_Js'][2]**2)
                        + c.INFO + ' J8: ' + c.NUMB + '{:.2f}'.format((ToF.Js[4]-ToF.opts['Target_Js'][3])**2/ToF.opts['Sigma_Js'][3]**2)
                     + c.ENDC)


                if self.opts['early stopping'] and new_cost < self.opts['optimum reached limit']:
                        print(c.INFO + 'Optimum achieved. Terminating.' + c.ENDC)
                        break   

                if abs(new_cost/old_cost-1) < self.opts['convergence limit']:
                    self.convergence_strikes += 1
                else:
                    self.convergence_strikes = 0
                #TODO: Redo
                if False and self.convergence_strikes >= 3 and ToF.opts['order'] == 4 and self.opts['improve order']:
                    self.convergence_strikes = 0
                    ToF.opts['order'] = 7
                    ToF.ss = (ToF.opts['order']+1)*[np.zeros(ToF.opts['N'])]
                    self.ToF_convergence_tolerances = max(self.opts['ToF convergence tolerances'][1], self.minimum_tolerance)
                    if self.opts['verbosity'] > 0: print(c.INFO + 'Approaching optimal solution. Increasing ToF order to:' + c.NUMB + str(ToF.opts['order']) + c.ENDC)

                if False and self.convergence_strikes >= 2 and ToF.opts['order'] == 7 and self.opts['improve order']:
                    self.convergence_strikes = 0
                    ToF.opts['order'] = 10
                    ToF.ss = (ToF.opts['order']+1)*[np.zeros(ToF.opts['N'])]
                    self.ToF_convergence_tolerances = max(self.opts['ToF convergence tolerances'][2], self.minimum_tolerance)
                    if self.opts['verbosity'] > 0: print(c.INFO + 'Approaching optimal solution. Increasing ToF order to:' + c.NUMB + str(ToF.opts['order']) + c.ENDC)

                if self.opts['early stopping'] and self.convergence_strikes >= 3:
                        print(c.INFO + 'Convergence detected. Terminating.' + c.ENDC)
                        break              

                if False and self.opts['verbosity'] > 3:
                    print(ToF.Js)
                    print(ToF.opts['Target_Js'])

                if random.random() < self.opts['kitty'] and self.opts['verbosity'] > 0:

                    self.huh()

                tic = time.perf_counter()

                #====================================== MAIN OPTIMISATION LOOP ======================================

                for step in range(steps):


                    params = opt_step(AdamOptimiser, self, ToF, params)

                    """
                    Js, Mass = model(OptToF, ToF)
                    loss = loss(OptToF, ToF, Js, Mass)
                    loss.backward()
                    opt.step()

                    if epoch < 5:
                        params = opt_step(AdamOptimiser, self, ToF, params)
                    else:
                        params = terrible_step(AdamOptimiser, self, ToF, params)

                    if epoch < 100:
                        params = opt_step(AdamOptimiser, self, ToF, params)
                    else:
                        params = mcterrible_step(AdamOptimiser, self, ToF, params)

                    explanation_strength = (ToF.Js[1:] - ToF.opts['Target_Js'])**2/ToF.opts['Sigma_Js']**2
                    if explanation_strength[0] > 1:
                        params = opt_step(AdamOptimiser, self, ToF, params)
                    elif explanation_strength[1] > 1:
                        params = opt_step(AdamOptimiser, self, ToF, params)
                    elif explanation_strength[2] > 1:
                        params = opt_step(AdamOptimiser, self, ToF, params)
                    elif explanation_strength[3] > 1:
                        params = opt_step(AdamOptimiser, self, ToF, params)
                    else:
                        break
                    """
                    cost = calc_cost(ToF)
                    self.cost_vector.append(calc_cost(ToF))
                    self.improvement_running_average = self.update_running_average(self.improvement_running_average, self.cost_vector[-1]/self.cost_vector[-2])

                    #precision increase
                    if self.opts['improve precision'] and cost < self.opts['precision increase cost threshold'] and not precision_fudge_factor < 2:
                        precision_fudge_factor /= 10
                        ToF.opts['Sigma_Js'] /= 10
                        if self.opts['verbosity'] > 1: print(c.INFO + 'Precision increased!' + c.ENDC)

                    if False and epoch == 2:
                        AdamOptimiser.learning_rate# = 1e-5


                    if False and epoch > 4 and cost < 9e3 and done == False:
                        #plt.plot(param_to_rho_exp_fixed(self, ToF, params))
                        #plt.show()
                        #AdamOptimiser.__init__()
                        AdamOptimiser.learning_rate = 1e-6
                        #ToF.opts['order'] = 7
                        #self.ToF_convergence_tolerance      = max(1e-12, 0.1*min(ToF.opts['Sigma_Js']))
                        #ToF.ss = (ToF.opts['order']+1)*[np.zeros(ToF.opts['N'])]
                        #print('=================================================')
                        done = True

                #====================================== MAIN OPTIMISATION LOOP ======================================

                toc = time.perf_counter()

                #TODO: Redo
                #lower learning rate
                if False and epoch/epochs > self.learning_rate_halvings_done/self.learning_rate_halvings: # and not self.improvement_running_average < 1 - self.opts['improvement leniency limit']/50
                    self.learning_rate /= 2
                    self.learning_rate_halvings_done += 1
                    AdamOptimiser.learning_rate = self.learning_rate
                    if self.opts['verbosity'] > 0: print(c.INFO + 'Learning rate halved. New learning rate is now: ' + c.NUMB + '{:.2e}'.format(self.learning_rate) +c.ENDC)
                
                #otherwise worsen learning rate
                elif False and abs(self.improvement_running_average-1) < self.opts['improvement strictness limit']:
                    self.learning_rate *= 2
                    self.learning_rate_halvings_done -= 1
                    AdamOptimiser.learning_rate = self.learning_rate
                    if self.opts['verbosity'] > 0: print(c.INFO + 'Learning rate doubled. New learning rate is now: ' + c.NUMB + '{:.2e}'.format(self.learning_rate) +c.ENDC)

                #tighten cost factor if it goes beyond the generous leniency limit of what is acceptable
                if not (1/self.opts['mass leniency limit'] < self.mass_fix_factor_running_average < self.opts['mass leniency limit']):
                    self.costfactor *= 10
                    if self.opts['verbosity'] > 0: print(c.INFO + 'Cost factor increased. New cost factor is now: ' + c.NUMB + '{:.2e}'.format(self.costfactor) +c.ENDC)

                #otherwise loosen cost factor if it stays within the very strict limit of what is "too good"
                elif 1/self.opts['mass strictness limit'] < self.mass_fix_factor_running_average < self.opts['mass strictness limit']:
                    self.costfactor /= 10
                    if self.opts['verbosity'] > 0: print(c.INFO + 'Cost factor decreased. New cost factor is now: ' + c.NUMB + '{:.2e}'.format(self.costfactor) +c.ENDC)

            ax2.plot(param_to_rho_exp_fixed(self, ToF, params))

            if (self.opts['verbosity'] > 0):

                total_stop_time = time.perf_counter()

                print(c.INFO + 'Final cost: ' + c.NUMB + '{:.2e}'.format(calc_cost(ToF)) + c.ENDC)

                print(c.INFO + 'Final Js: ' + c.ENDC)
                print(ToF.Js[1:])
                print(c.INFO + 'Target Js: ' + c.ENDC)
                print(ToF.opts['Target_Js'])
                print(c.INFO + 'Difference: ' + c.ENDC)
                print(ToF.Js[1:]-ToF.opts['Target_Js'])
                print(c.INFO + 'Tolerance: ' + c.ENDC)
                print(ToF.opts['Sigma_Js'])

                print()

                if self.opts['time']:
                    print(c.INFO + 'Total time: ' + c.NUMB + '{:.2f}'.format(total_stop_time-total_start_time) + c.INFO + ' seconds.' +c.ENDC)
                    print(c.INFO + 'Time for gradient precalculations: ' + c.NUMB + '{:.2f}'.format(self.timing[0]) + c.INFO + ' seconds.' +c.ENDC)
                    print(c.INFO + 'Time for ToF calculations: ' + c.NUMB + '{:.2f}'.format(self.timing[1]) + c.INFO + ' seconds.' +c.ENDC)
                    print(c.INFO + 'Time for gradient calculations: ' + c.NUMB + '{:.2f}'.format(self.timing[2]) + c.INFO + ' seconds.' +c.ENDC)

                #print(cost_vector)
                ax3.semilogy(np.array(self.cost_vector))
                plt.show()

            return params
    

    def update_running_average(self, average, new):
        return self.opts['rolling average forgetfulness']*average + (1-self.opts['rolling average forgetfulness'])*new

    def update_mass_fix_factor_running_average(self, mass_fix_factor):
        self.mass_fix_factor_running_average = (self.opts['rolling average forgetfulness']*self.mass_fix_factor_running_average + (1-self.opts['rolling average forgetfulness'])*mass_fix_factor)

    def huh(self):
        print(c.INFO + 'A confused little kitten has stumbled onto the terminal!' + c.ENDC)
        print()
        print()
        print('                                                     ／l、             ')
        print('                                                   （ﾟ､ ｡ ７         ')
        print('                                                     |   ~ヽ       ')
        print('                                                     じしf_,)ノ')
        print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww')
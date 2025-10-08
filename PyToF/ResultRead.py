import numpy as np
import h5py
import time
import signal

from color import c

"""
Template file to read results from the hdf5 file using h5py

file structure:

read_file():                    main function to call that reads file
interrupt_handler():            helper function to interrupt result generation. check keeprunning in your loop
read_datasets(f):               given a file, generates a dictionary results[] with all the results you want in it
analyse_dataset(name, object):  function visited within generate results per dataset that analyses that dataset and adds the found attributes to the results vectors
                                analyse dataset must share a lot of variables with read_datasets (since it writes to them) but due to visititems requirements, must keep its signature
                                if one could be bothered, one could wrap this in a class or, or, one simply declares all relevant variables as globals (guess which one we do)
initialisation and call of read_file()
"""

def read_file():
    """
    Main file reading function. Manipulation of all results together goes here (plotting etc).
    """

    tic = time.perf_counter()

    #neptune
    if True:
        filename = 'bigrun_neptune.hdf5'
    #uranus
    else:
        filename = 'bigrun_uranus.hdf5'

    #this with is load-bearing because it neatly closes the filestream after we leave scope. close it manually if you dont want to use with
    with h5py.File(filename, 'r') as f:
        results = read_datasets(f)

    toc = time.perf_counter()

    print(c.INFO + "Total time to read file:  " + c.NUMB + time.strftime("%M:%S", time.gmtime((toc-tic))) + c.ENDC)

    return

def interrupt_handler(sig, frame):
    global keeprunning
    keeprunning = False
    print(c.WARN + 'Termination request raised.' + c.ENDC)

def read_datasets(f):
    """
    Interface function. Not necessary and can be combined with read_file, but wraps the messy global hack into a nice result dict.
    Wrapping of results into dict goes here.
    Returns a dictionary containing certain results read from the hdf5 file supplied. Structure is:
    results: result dictionary. indexed as results['attribute'] for the following:
    example:                        INT number of datasets visited
    """

    #set the interrupt handler at this level
    signal.signal(signal.SIGINT, interrupt_handler)

    global example

    example = 0

    #main loop
    # note we use visititems because of the significant speed gain.
    # the disadvantage is that visititems must be given a callable (function) of exact signature callable(name, object)
    # if callable returns None, the visiting continues. Returning value will immediately stop visiting and return that value
    # note: because this callable has exact signature, no extra variables can be explicitly passed.
    #       Your options are:
    #           use global variables (easy and works but is mcterrible™) 
    #           wrap it in a class (untested but should work)
    #           some other great option I havent thought of
    # note: this CANNOT be multithreaded
    # note: if you wish to time the performance of callable, some hidden time will be obscured within the visititems process.
    #        darktime measures that. She's also very ominous and loves to loom in the shadowiest booth of the tavern.

    timestartcall = time.perf_counter()
    timeendcall = time.perf_counter()
    darktime = 0
    f.visititems(analyse_dataset)

    results = {}

    results['example'] = example

    return results

def analyse_dataset(name, object):

    """
    Function called once per dataset in the hdf5 file. By default, analyse_dataset has no memory and is re-initialised for each dataset.
    It would be wise to ensure large objects that are used repeatedly are not initialised every time analyse_dataset is called.
    When called by visititem, analyse_dataset is given the name fo the dataset (its key, I chose a UUID), and the object itself (i.e. the actual data).
    Data is formatted as explained 10 lines below.
    Actual work to be done for each resulting distribution goes here.
    """

    global example
    global timestartcall, timeendcall, darktime

    timestartcall = time.perf_counter()
    darktime += timestartcall - timeendcall #time between start of this call and end of last call, i.e., time used by visititems

    #the filestructure of the hdf5 file. each dataset is composed of two arrays. Array at object[0] is the starting density distribution, object[1] the result density distribution.
    #attributes contain rho_MAX respected (if rho_MAX was respected) and Js explained (if the Js were explained by the result)
    #the arrays behave like numpy arrays
    starting_rho = object[0]
    rho = object[1]
    rhoexplained = object.attrs['rho_MAX respected']
    Jsexplained = object.attrs['Js explained']

    #example (trivial) dataset analysis
    example += 1

    #progress tracker (sort of like tqdm)
    if example % 100 == 0:
        duration = time.perf_counter() - starttime
        rate = (example/duration)
        esttimerem = (1341762 - example)/rate
        print('  Nr: ' + str(example) +'. Time since start: ' + time.strftime("%H:%M:%S", time.gmtime(duration)) + '. Rate: ' + '{:.0f}'.format(rate) + ' it/s. Estimated time remaining: ' + time.strftime("%H:%M:%S", time.gmtime(esttimerem)) + '  ' , end='\r') #end='\r' ensures it doesnt spam the console but replaces the same line

    if keeprunning == False: return "Interrupted" #anything except None here

    timeendcall = time.perf_counter()

    return None

# the actual call

starttime = time.perf_counter()
keeprunning = True
read_file()
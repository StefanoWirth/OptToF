import numpy as np
import h5py
import time
import pickle
import signal
from ClassToF import ToF
import AlgoToF
from FunctionsToF import get_NMoI, _pressurize

N_uranus = 1277081
N_neptune = 1341762

rho_max = 2e4
p_max = 3e12

#IMPORTANT NOTICE: ensure files 'bigrun_neptune.hdf5' and 'bigrun_uranus.hdf5' are in the same directory, as are ClassToF, AlgoToF, FunctionsToF.
is_neptune = False
dodifferentjumpcriteria = True

def save_results():
    tic = time.perf_counter()
    global jumpcriterion, jumpcriteria
    jumpcriterion = 100
    #jumpcriteria = [50,75,100,150,250,500,1000]
    jumpcriteria = np.arange(50, 1001, 50)

    #neptune
    if is_neptune:
        filename = 'newrun_neptune.hdf5'
    #uranus
    else:
        filename = 'newrun_uranus.hdf5'


    with h5py.File(filename, 'r') as f:
        results = generate_results(f)


    if is_neptune:
        with open('result_dict_neptune.pkl', 'xb') as f:
            pickle.dump(results, f)
    else:
        with open('result_dict_uranus.pkl', 'xb') as f:
            pickle.dump(results, f)

    toc = time.perf_counter()

    print("Total time to generate results:  " + time.strftime("%M:%S", time.gmtime((toc-tic))))
    print("Of which file handling:          " + time.strftime("%M:%S", time.gmtime(darktime + timing[0])))
    print("Of which dataset analysing:      " + time.strftime("%M:%S", time.gmtime(timing[1] + timing[2] + timing[3] + timing[4] + timing[5])))
    print()
    print("Number of datasets analysed:     " + str(results['n']))
    print("Number of which respect rho_MAX: " + str(results['nR']))
    print("Number of which explain Js:      " + str(results['nJ']))
    print("Percentage respect rho_MAX:      " + '{:.2%}'.format(results['rho_MAX respected%']))
    print("Percentage explained Js:         " + '{:.2%}'.format(results['Js explained%']))

    return

def interrupt_handler(sig, frame):
    global keeprunning
    keeprunning = False
    print('Termination request raised.')

def generate_results(f):

    """
    Returns a dictionary containing certain results read from the hdf5 file supplied. Structure is:
    results: result dictionary. indexed as results['attribute'] for the following:
    n:                          INT number of samples
    nR:                         INT number of samples that respected rho_MAX
    nJ:                         INT number of samples that respected rho_MAX and explained J
    rho_MAX respected%:         FLOAT !NOT PERCENTAGES ratio of result densities that did not exceed rho_MAX
    Js explained%:              FLOAT !NOT PERCENTAGES ratio of result densities that did not exceed rho_MAX that explained the Js
    min dens:                   ARRAY (from list) minimum densities of result distributions that respect rho_MAX
    max dens:                   ARRAY (from list) maximum densities of result distributions that respect rho_MAX
    max start dens:             FLOAT average maximum starting density over all starting distributions
    moments of inertia:         ARRAY (from list) moments of inertia of result distributions that explain the Js
    flattening ratios:          ARRAY (from list) flattening coefficient of result distributions that explain the Js
    dens Js explained view:     ARRAY (from list) of bools that creates a view for only runs that explained the Js 
    raw jumps tot:              ARRAY of shape rho containing the jump increase over all detected jumps at that step (over all rho that respect rho_MAX, divided by number of such)
    raw jumps expl:             ARRAY of shape rho containing the jump increase over all detected jumps at that step (over all rho that respect rho_MAX and explain Js, divided by number of such)
    raw jumps not expl:         ARRAY of shape rho containing the jump increase over all detected jumps at that step (over all rho that respect rho_MAX and don't explain Js, divided by number of such)
    processed jumps:            2D ARRAY (from list of tuple) containing real jumplocation as float, jumpsize (dx) and jumpmagnitude (dy) (over all that respect rho_MAX)
    jumps Js explained view:    ARRAY (from list) of bools that creates a view of processed jumps for only those that explained the Js
    nr jumps:                   ARRAY (from list) number of jumps of result distributions that respect rho_MAX
    nr jumps Js explained view: ARRAY (from list) of bools that creates a view of nr jumps for only those that explained the Js 
    nr jumps per criterion:     2D ARRAY (from list of lists) of shape len(jumpcriteria) * number of samples that tracks number of jumps of result distributions that respect rho_MAX over some possible jumpcriteria
    NOPE rho_MAX failures:      NOPE     2D ARRAY of all failure starting distributions (rho_MAX not respected)
    NOPE J failures:            NOPE     2D ARRAY of all failure starting distributions (rho_MAX respected but Js not explained)
    avg start:                  ARRAY of shape rho containing the average starting distribution
    avg respected start:        ARRAY of shape rho containing the average starting distribution respected rho_MAX
    avg successful start:       ARRAY of shape rho containing the average starting distribution that ended up explaining the Js
    avg successful result:      ARRAY of shape rho containing the average resulting distribution that ended up explaining the Js
    avg change:                 ARRAY of shape rho containing the average change from starting distribution to distribution that explained the Js
    x:                          ARRAY of shape N, x coordinate
    y:                          ARRAY of shape RES, y coordinate
    RES:                        INT resolution
    distr grid density:         2D ARRAY of shape N (x, first direction) * RES (y, second direction, [0, rho_MAX]) that bins the location of the rho onto a coordinate grid. This results in a probability distribution of sorts.
    distr grid pressure:        2D ARRAY of shape N (x, first direction) * RES (y, second direction, [0, rho_MAX]) that bins the location of the pressures onto a coordinate grid. This results in a probability distribution of sorts.
    """

    signal.signal(signal.SIGINT, interrupt_handler)

    global timing, jumptiming, toftiming

    timing = [0,0,0,0,0,0,0] #fileread, dens, avgchange, ToF (calc, MoI, ratios, pressures), distr, jump, jumpcriteria   
    jumptiming = [0,0,0,0,0,0,0] # find jump | cut fat | find location | save results | wasted | initial gen | searching
    toftiming = [0,0,0] #algotof | moments of inertia | pressure

    global N, n, nrespected, nJsexplained
    global min_dens, max_dens, max_start_dens, dens_Js_explained_view
    global moments_of_inertia, flattening_ratios
    global raw_jumps_tot, raw_jumps_expl, raw_jumps_not_expl, processed_jumps, jumps_Js_explained_view, nr_jumps, nr_jumps_Js_explained_view, nr_jumps_per_criterion
    global rho_MAX_failures, J_failures
    global avg_start, avg_respect_start, avg_successful_start, avg_successful_result, avg_change
    global RES, x, y, distr_grid_dens, distr_grid_pressure
    
    N = 2**10    #size of datasets
    n = 0
    nrespected = 0
    nJsexplained = 0

    min_dens = []
    max_dens = []
    max_start_dens = 0
    dens_Js_explained_view = []

    moments_of_inertia = []
    flattening_ratios = []

    raw_jumps_tot = np.zeros(N - 1)
    raw_jumps_expl = np.zeros(N - 1)
    raw_jumps_not_expl = np.zeros(N - 1)
    processed_jumps = []
    jumps_Js_explained_view = []
    nr_jumps = []
    nr_jumps_Js_explained_view = []
    nr_jumps_per_criterion = [[] for criterion in jumpcriteria]

    rho_MAX_failures = []
    J_failures = []

    avg_start = np.zeros(N)
    avg_respect_start = np.zeros(N)
    avg_successful_start = np.zeros(N)
    avg_successful_result = np.zeros(N)
    avg_change = np.zeros(N)

    RES = 4*N
    x = np.arange(N)
    y = np.linspace(0, rho_max, RES)
    distr_grid_dens = np.zeros((N, RES))
    distr_grid_pressure = np.zeros((N, RES))

    #main loop
    global timestartcall, timeendcall, darktime
    timestartcall = time.perf_counter()
    timeendcall = time.perf_counter()
    darktime = 0
    f.visititems(analyse_dataset)


    results = {}

    results['n']  = n
    results['nR'] = nrespected
    results['nJ'] = nJsexplained
    results['rho_MAX respected%'] =  nrespected / n
    results['Js explained%'] = nJsexplained / nrespected

    results['min dens'] = np.array(min_dens)
    results['max dens'] = np.array(max_dens)
    results['max start dens'] = max_start_dens / n
    results['dens Js explained view'] = np.array(dens_Js_explained_view)
  
    results['moments of inertia'] = np.array(moments_of_inertia)
    results['flattening ratios'] = np.array(flattening_ratios)

    results['raw jumps tot'] = raw_jumps_tot / nrespected
    results['raw jumps expl'] = raw_jumps_expl / nJsexplained
    results['raw jumps not expl'] = raw_jumps_not_expl / (nrespected - nJsexplained)
    results['processed jumps'] = np.asarray(processed_jumps)
    results['jumps Js explained view'] = np.array(jumps_Js_explained_view)
    results['nr jumps'] = np.array(nr_jumps)
    results['nr jumps Js explained view'] = np.array(nr_jumps_Js_explained_view)
    results['nr jumps per criterion'] = np.asarray(nr_jumps_per_criterion)

    results['rho_MAX failures'] = np.asarray(rho_MAX_failures)
    results['J failures'] = np.asarray(J_failures)

    results['avg start'] = avg_start / n
    results['avg respected start'] = avg_respect_start / nrespected
    results['avg successful start'] = avg_successful_start / nJsexplained
    results['avg successful result'] = avg_successful_result / nJsexplained
    results['avg change'] = avg_change / nJsexplained

    results['x'] = x
    results['y'] = y
    results['RES'] = RES
    results['distr grid density'] = np.divide(distr_grid_dens, nJsexplained)
    results['distr grid pressure'] = np.divide(distr_grid_pressure, nJsexplained)

    print()
    #fileread, dens, avgchange, ToF (calc, MoI, ratios, pressures), distr, jump, jumpcriteria
    print('Time to find files:              ' + time.strftime("%M:%S", time.gmtime((darktime))))
    print('Time to read files:              ' + time.strftime("%M:%S", time.gmtime((timing[0]))))
    print('Time for density:                ' + time.strftime("%M:%S", time.gmtime((timing[1]))))
    print('Time for average change:         ' + time.strftime("%M:%S", time.gmtime((timing[2]))))
    print('Time for ToF, MoI, pressure:     ' + time.strftime("%M:%S", time.gmtime((timing[3]))))
    print('Time for distribution:           ' + time.strftime("%M:%S", time.gmtime((timing[4]))))
    print('Time for jumps:                  ' + time.strftime("%M:%S", time.gmtime((timing[5]))))
    print('Time for jumpcriteria:           ' + time.strftime("%M:%S", time.gmtime((timing[6]))))
    # find jump | cut fat | find location | save results | wasted | initial gen | searching
    print('Time to find jump length:        ' + time.strftime("%M:%S", time.gmtime((jumptiming[0]))))
    print('Time to cut fat:                 ' + time.strftime("%M:%S", time.gmtime((jumptiming[1]))))
    print('Time to find location:           ' + time.strftime("%M:%S", time.gmtime((jumptiming[2]))))
    print('Time to save result:             ' + time.strftime("%M:%S", time.gmtime((jumptiming[3]))))
    print('Time spent on false jumps:       ' + time.strftime("%M:%S", time.gmtime((jumptiming[4]))))
    print('Time spent on initial generation:' + time.strftime("%M:%S", time.gmtime((jumptiming[5]))))
    print('Time to find jumps:              ' + time.strftime("%M:%S", time.gmtime((jumptiming[6]))))
    #algotof | moments of inertia | pressure
    #print('Time spent on AlgoToF:           ' + time.strftime("%M:%S", time.gmtime((toftiming[0]))))
    #print('Time spent on MoI:               ' + time.strftime("%M:%S", time.gmtime((toftiming[1]))))
    #print('Time to pressurise:              ' + time.strftime("%M:%S", time.gmtime((toftiming[2]))))

    return results

def analyse_dataset(name, object):

    global timing, jumptiming, toftiming

    global N, n, nrespected, nJsexplained
    global min_dens, max_dens, max_start_dens, dens_Js_explained_view
    global moments_of_inertia, flattening_ratios
    global raw_jumps_tot, raw_jumps_expl, raw_jumps_not_expl, processed_jumps, jumps_Js_explained_view, nr_jumps, nr_jumps_Js_explained_view, nr_jumps_per_criterion
    global rho_MAX_failures, J_failures
    global avg_start, avg_respect_start, avg_successful_start, avg_successful_result, avg_change
    global RES, x, y, distr_grid_dens, distr_grid_pressure
    
    global timestartcall, timeendcall, darktime
    timestartcall = time.perf_counter()
    darktime += timestartcall - timeendcall #time between start of this call and end of last call, i.e., time used by visititems

    time0 = time.perf_counter()

    starting_rho = object[0]
    rho = object[1]
    pressure = object[2]
    rhoexplained = object.attrs['rho_MAX respected']
    Jsexplained = object.attrs['Js explained']
    flattening_ratio = object.attrs['flattening ratio']
    nmoi = object.attrs['nmoi']

    time1 = time.perf_counter()

    n += 1
    avg_start += starting_rho

    if rhoexplained == False:
        #rho_MAX_failures.append(starting_rho)
        return None

    nrespected += 1

    if Jsexplained == True: nJsexplained += 1
    #else: J_failures.append(starting_rho)

    #dens
    #assert(rho[1]>0)
    min_dens.append(rho[1])
    max_dens.append(rho[-1])
    max_start_dens += starting_rho[-1]
    dens_Js_explained_view.append(Jsexplained)

    time2 = time.perf_counter()

    avg_respect_start += starting_rho

    if Jsexplained == True:
        #avg start/result & change
        avg_successful_start += starting_rho
        avg_successful_result += rho
        avg_change += rho - starting_rho

    time3 = time.perf_counter()

    #flattening ratios | moments of inertia | pressure

    if Jsexplained == True:
        flattening_ratios.append(flattening_ratio)
        moments_of_inertia.append(nmoi)

    time4 = time.perf_counter()

    if Jsexplained == True:
        #distribution
        distr_grid_dens[np.arange(N), np.floor_divide(rho * RES, rho_max, casting='unsafe' ,dtype=np.dtype(int))] += 1
        distr_grid_pressure[np.arange(N), np.minimum(np.floor_divide(pressure * RES, p_max, casting='unsafe' ,dtype=np.dtype(int)), RES - 1)] += 1
        #the amount of obscure wizardry performed in one singular line here is likely hitherto unparalleled
        # actually its just indexing over all x and the y index rho * RES // max_RHO (linear interpolation from 0 to max_RHO) with a safeguard
    time5 = time.perf_counter()

    #jumps
    """
    Good Luck.

    Our task is to find every jump of any arbitrary size
    A jumplet is defined as an interval of size k such that the total increase over that interval is more than k*jumpcriterion
    That is, the total average increase over this interval is more than jumpcriterion
    A jump is defined as the largest jumplet of a grouping, that is, given a cluster of jumplets that intersect one another, there exists exactly one jumplet that contains all the others. This is the jump.
    This is true because if the jumplets intersect one another, then the average over their size is greater than jumpcriterion, thus also the larger average.
    Example:
    101, 99, 101 is one jump because [1, 1], [3, 3], and [1, 3] are jumplets.
    100, 0, 100 are two jumps because [1, 3] is not a jumplet.
    Naïvely, one could iterate over all jumpsizes and find the jumps that way. That is bad.
    We thus use an approximate solution using a sliding window.  
    if we used an averaging sliding window (i.e., [0.25, 0.25, 0.25, 0.25]), each one will only guarantee to find jumps of size of the window
    this is because smaller jumps get squashed by the factor 0.25 and larger jumps could erroneously be split
    for example: sliding window size 2: 125 125 0 125 125 is 1 jump (average 100) but gets seen as 2.
    If we used a constant sliding window (i.e. [1 1 1 1]), each one will hugely overaccount for too small of an increase (especially at large window sizes).
    therefore we use a combination of different window sizes at the same time
    we scale this combination as low as possible to guarantee we will still find all jumps of size up to this window:
    it must certainly contain one entry >= 1 to find 1-jumps, two entries >= 0.5 to find 2-jumps, etc.
    up to 9-jumps, which requires nine entries >= 1/9
    because it is highly likely that jumps larger than this window will have subsections within this window that are captured, this essentially guarantees all jumps are found.
    this ensures that a jump even of only size 1 will certainly at some point exceed the jumpcriterion
    but also, a jump spread over 9 steps will show up

    Because this method is still quite a bit too generous (similarly to the problem with large sliding windows), we must then cut down the jump as aggressively as possible and check if it qualifies.
    """
    tic = time.perf_counter()
    #roll array for roll calculation (see explanation above)
    #roll = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    roll = np.array([1/8, 1/6, 1/4, 1/2, 1, 1/3, 1/5, 1/7, 1/9])
    diffs = np.diff(rho)
    rolled = np.convolve(diffs, roll, mode='same')
    isjump = (rolled > jumpcriterion)
    i = 0
    nr_jumps_this_dataset = 0
    jumptiming[5] += time.perf_counter() - tic

    while i < len(rolled): #go through the rolling average

        jumptimeminus1 = time.perf_counter()

        #if, at some point, the weighted window of size 9 (probably at the center), exceeds the total jumpsize (rescaled by 1/9th)
        nextjump = isjump[i:].argmax() #go through the sliced bool array that is true where rolled is bigger than jumpcriterion
        i += nextjump

        if nextjump == 0: #this is either because we really already were at a jumppoint (should happen only at the start but also there never)
                          #OR importantly, because there are no more jumps left (in which case isjump[i] would be FALSE)
            if isjump[i] == False:
                #we're done, no more jumps
                jumptiming[6] += time.perf_counter() - jumptimeminus1
                break

        jumptime0 = time.perf_counter()
        jumptiming[6] += jumptime0 - jumptimeminus1

        jumpstart = i
        # we found a candidate jumpstart at i
        while i < len(rolled) and rolled[i] > jumpcriterion:
            i += 1
        jumpend = i
        # found the candidate's jumpend at i
        #we've found a jump. cut fat from below then above, affirm the leapsize is viable and find its center
        #while cutting fat would improve the jumpratio (jumpmagnitude/jumpsize) and is also less than the jumpmagnitude
        jumptime1 = time.perf_counter()
        k = 0 #how much to cut from below
        while jumpstart + k + 1 < jumpend and diffs[jumpstart + k] < jumpcriterion and (rho[jumpend]-rho[jumpstart + k + 1])/(jumpend - (jumpstart + k + 1)) > (rho[jumpend]-rho[jumpstart + k])/(jumpend - jumpstart - (k)):
            k += 1
        jumpstart = jumpstart + k
        #we cut fat up to jumpstart + k
        j = 0 #how much to cut from above
        while jumpstart < jumpend - j - 1 and diffs[jumpend - 1 - j] < jumpcriterion and (rho[jumpend - j - 1]-rho[jumpstart])/((jumpend - j - 1) - jumpstart) > (rho[jumpend - j]-rho[jumpstart])/((jumpend - j) - jumpstart):
            j += 1
        jumpend = jumpend - j
        i = jumpend
        # we cut fat down to jumpend - j
        jumptime2 = time.perf_counter()

        """
        if jumpstart >= jumpend:
            # just to be sure
            continue
        """
        jumpmagnitude = rho[jumpend]-rho[jumpstart]
        jumpsize = jumpend - jumpstart
        if jumpmagnitude / jumpsize < jumpcriterion:
            jumptiming[4] += time.perf_counter() - jumptimeminus1
            # viability check
            continue
        jumplocation = 0
        jumpsteps = diffs[jumpstart:jumpend] #excludes end index
        """
        #UNNECESSARY:
        if max(jumpsteps) < jumpcriterion:
            # sanity check
            print("WAH2")
            continue
        """
        for l in range(len(jumpsteps)):
            jumplocation += l*jumpsteps[l]
        jumplocation /= sum(jumpsteps)
        real_jumplocation = jumpstart + jumplocation

        jumptime3 = time.perf_counter()

        """
        raw jumps tot:          ARRAY of shape rho - 1 containing the jump increase over all detected jumps at that step (over all rho that respect rho_MAX, divided by number of such)
        raw jumps expl:         ARRAY of shape rho - 1 containing the jump increase over all detected jumps at that step (over all rho that respect rho_MAX and explain Js, divided by number of such)
        raw jumps not expl:     ARRAY of shape rho - 1 containing the jump increase over all detected jumps at that step (over all rho that respect rho_MAX and don't explain Js, divided by number of such)
        processed jumps:        2D ARRAY (from list of tuple) containing real jumplocation as float, jumpsize (dx) and jumpmagnitude (dy) (over all that respect rho_MAX)
        jumps Js explained view:ARRAY (from list) of bools that creates a view of processed jumps for only those that explained the Js
        """

        raw_jumps_tot[jumpstart:jumpend] += jumpsteps
        if Jsexplained == True:
            raw_jumps_expl[jumpstart:jumpend] += jumpsteps
        else:
            raw_jumps_not_expl[jumpstart:jumpend] += jumpsteps
        processed_jumps.append((real_jumplocation, jumpsize, jumpmagnitude))
        jumps_Js_explained_view.append(Jsexplained)
    
        nr_jumps_this_dataset += 1

        jumptime4 = time.perf_counter()
        
        # find jump len | cut fat | find location | save results | wasted | initial gen | searching
        jumptiming[0] += jumptime1 - jumptime0
        jumptiming[1] += jumptime2 - jumptime1
        jumptiming[2] += jumptime3 - jumptime2
        jumptiming[3] += jumptime4 - jumptime3

    nr_jumps.append(nr_jumps_this_dataset)
    nr_jumps_Js_explained_view.append(Jsexplained)

    time6 = time.perf_counter()

    #jumpcriteriaplot

    #now we do this all again for sport for all Jsexplained
    #this is all the same code copied, redundant parts gutted, one timing, no comments, tightened
    if Jsexplained == True and dodifferentjumpcriteria == True:
        for s in range(len(jumpcriteria)):
            jumpcriterion_ = jumpcriteria[s]
            isjump = (rolled > jumpcriterion_)
            i = 0
            nr_jumps_this_dataset_ = 0
            while i < len(rolled):
                nextjump = isjump[i:].argmax()
                i += nextjump
                if nextjump == 0:
                    if isjump[i] == False:
                        break
                jumpstart = i
                while i < len(rolled) and rolled[i] > jumpcriterion_:
                    i += 1
                jumpend = i
                k = 0
                while jumpstart + k + 1 < jumpend and diffs[jumpstart + k] < jumpcriterion_ and (rho[jumpend]-rho[jumpstart + k + 1])/(jumpend - (jumpstart + k + 1)) > (rho[jumpend]-rho[jumpstart + k])/(jumpend - jumpstart - (k)):
                    k += 1
                jumpstart = jumpstart + k
                j = 0
                while jumpstart < jumpend - j - 1 and diffs[jumpend - 1 - j] < jumpcriterion_ and (rho[jumpend - j - 1]-rho[jumpstart])/((jumpend - j - 1) - jumpstart) > (rho[jumpend - j]-rho[jumpstart])/((jumpend - j) - jumpstart):
                    j += 1
                jumpend = jumpend - j
                i = jumpend
                if (rho[jumpend]-rho[jumpstart]) / (jumpend - jumpstart) < jumpcriterion_:
                    continue
                nr_jumps_this_dataset_ += 1
            nr_jumps_per_criterion[s].append(nr_jumps_this_dataset_)

    time7 = time.perf_counter()

    timing[0] += time1 - time0
    timing[1] += time2 - time1
    timing[2] += time3 - time2
    timing[3] += time4 - time3
    timing[4] += time5 - time4
    timing[5] += time6 - time5
    timing[6] += time7 - time6

    if n % 100 == 0:
        duration = time.perf_counter() - starttime
        rate = (n/duration)
        esttimerem = (1341762 - n)/rate
        days = int(esttimerem // (60*60*24))
        esttimerem = esttimerem%(60*60*24)
        print('  Nr: ' + str(n) +'. Time since start: ' + time.strftime("%H:%M:%S", time.gmtime(duration)) + '. Rate: ' + '{:.0f}'.format(rate) + ' it/s. Estimated time remaining: ' + str(days) + "d "+ time.strftime("%H:%M:%S", time.gmtime(esttimerem)) + '  ' , end='\r') 

    if keeprunning == False: return "Interrupted"

    timeendcall = time.perf_counter()

    return None

starttime = time.perf_counter()
keeprunning = True

save_results()

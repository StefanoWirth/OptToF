import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.transforms as trans
#np.set_printoptions(legacy='1.25')

N_uranus = 1277081
N_neptune = 1341762

N = 2**10    #size of datasets
RES = 4*N

is_neptune = False

def generate_plots():
    tic = time.perf_counter()
    global jumpcriterion, jumpcriteria
    jumpcriterion = 100
    jumpcriteria = [50,75,100,150,250,500,1000]


    if is_neptune:
        with open('result_dict_neptune.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        with open('result_dict_uranus.pkl', 'rb') as f:
            results = pickle.load(f)


    toc = time.perf_counter()



    #PLOT ================================

    fig, axs = plt.subplots(4, 4, layout = 'constrained', num = str(is_neptune))

    bincount = 200    #THE BINS MARTY THE BINS WHAT DO THEY MEAN

    nticks = 5

    xticklocations = np.linspace(1023, 0, nticks)
    xticklabels = np.linspace(0, 1, nticks)
    yticklocations = [0, 5000, 10000, 15000, 20000]
    yticklabels = [0, 5, 10, 15, 20]

    #JUMPS

    axs[0,0].set_title('Raw Jumps of J explained')
    axs[0,0].bar(range(1023), results['raw jumps expl'], width=1)
    axs[0,0].invert_xaxis()
    axs[0,0].set_xticks(xticklocations, xticklabels)
    axs[0,0].set_xlabel("Normalised radius $r/R$")


    """
    axs[0,1].set_title('Raw Jumps of J explained (averaged)')
    axs[0,1].plot(np.convolve(results['raw jumps expl'], np.ones(5)/5, mode = 'same'))
    """
    axs[0,1].set_title('Processed Jumps of J explained') #((real_jumplocation, jumpsize, jumpmagnitude))
    proc_jump_j_expl = results['processed jumps'][results['jumps Js explained view']]
    jumpindex = proc_jump_j_expl[:, 0]
    jumpwidth = proc_jump_j_expl[:, 1]
    jumpheight = proc_jump_j_expl[:, 2]
    jumpweight = jumpheight/jumpwidth
    axs[0,1].hist(jumpindex, bins = bincount, weights = jumpweight)
    axs[0,1].invert_xaxis()
    axs[0,1].set_xticks(xticklocations, xticklabels)
    axs[0,1].set_xlabel("Normalised radius $r/R$")

    axs[0,2].set_title('Processed Jumps of J explained\njumpsize') #((real_jumplocation, jumpsize, jumpmagnitude))
    axs[0,2].hist(jumpindex, bins = bincount, weights = jumpwidth)
    axs[0,2].invert_xaxis()
    axs[0,2].set_xticks(xticklocations, xticklabels)
    axs[0,2].set_xlabel("Normalised radius $r/R$")

    axs[0,3].set_title('Processed Jumps of J explained\njumpheight') #((real_jumplocation, jumpsize, jumpmagnitude))
    axs[0,3].hist(jumpindex, bins = bincount, weights = jumpheight)
    axs[0,3].invert_xaxis()
    axs[0,3].set_xticks(xticklocations, xticklabels)
    axs[0,3].set_xlabel("Normalised radius $r/R$")

    axs[1,2].set_title('Number of Jumps of J explained')
    nr_jumps_J_expl = results['nr jumps'][results['nr jumps Js explained view']]
    max_nr = np.max(nr_jumps_J_expl)
    nrjumpsbinvalues, nrjumpsbinedges, _ = axs[1,2].hist(nr_jumps_J_expl, bins = max_nr + 1, range = (-0.5, max_nr+0.5))
    #find index where we cumulate 95% of the total nr of jumps
    transform = trans.blended_transform_factory(axs[1,2].transData, axs[1,2].transAxes)
    axs[1,2].vlines(np.percentile(nr_jumps_J_expl, 95), ymin = 0, ymax = 1, linestyle ='--', color = 'tab:gray', transform = transform)
    
    axs[1,3].set_title('Average and 95% jumps')
    jumps_per_criteria = results['nr jumps per criterion']
    jumpcriteria # x           
    avg_jumps_per_criteria = [] # y1
    max_jumps_per_criteria = [] # y2
    for jumps in jumps_per_criteria:
        avg_jumps_per_criteria.append(np.mean(jumps))
        max_jumps_per_criteria.append(np.percentile(jumps, 95))        #find index which covers 95th percentile
    axs[1,3].plot(avg_jumps_per_criteria, color = 'tab:red')
    axs[1,3].plot(max_jumps_per_criteria, color = '0') #black
    axs[1,3].set_xticks(range(len(jumpcriteria)))
    axs[1,3].set_xticklabels(jumpcriteria)
    #print(jumpcriteria)
    #print(avg_jumps_per_criteria)
    #print(max_jumps_per_criteria)

    """
    plt.title('All the ones that failed to explain J')
    for rho in results['J failures']:
        plt.plot(rho, alpha = 0.1)
    """

    #DISTR & CONTOUR

    axs[1,0].set_title('Log Distribution')
    log_grid = results['distr grid density']
    logdistr = axs[1,0].imshow((np.transpose(log_grid)), cmap = 'plasma', norm = col.LogNorm(), aspect = 'auto', origin = 'lower', extent = (-0.5, N-0.5, -0.5, 2e4-0.5))
    fig.colorbar(logdistr, ax=axs[1,0])
    axs[1,0].invert_xaxis()
    axs[1,0].set_xticks(xticklocations, xticklabels)
    axs[1,0].set_xlabel("Normalised radius $r/R$")
    axs[1,0].set_ylim(top = 20000)
    axs[1,0].locator_params(axis='y', nbins=5)
    axs[1,0].set_yticks(yticklocations, yticklabels)
    axs[1,0].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")


    axs[1,1].set_title('Contour')
    percentile_grid = np.zeros_like(results['distr grid density'])
    contourintervals = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    cumulative_distr = np.cumsum(results['distr grid density'], axis=1)
    twolowersd      = -np.ones(N) #at 2.5%
    lowersd         = -np.ones(N) #at 16%
    contourmedian   = -np.ones(N) #at 50%
    uppersd         = -np.ones(N) #at 84%
    twouppersd      = -np.ones(N) #at 97.5%

    for x in range(N):
        level = 0
        y = 0
        while level < 10:
            while cumulative_distr[x, y] < 0.5-contourintervals[level]/2:
                percentile_grid[x, y] += level
                if cumulative_distr[x, y] > 0.025 and twolowersd[x] == -1: twolowersd[x] = y*2e4/RES
                if cumulative_distr[x, y] > 0.16  and lowersd[x] == -1: lowersd[x] = y*2e4/RES
                y += 1
            level += 1
        level -= 1
        while level > -1:
            while cumulative_distr[x, y] < 0.5+contourintervals[level]/2:
                percentile_grid[x, y] += level + 1
                if cumulative_distr[x, y] > 0.5   and contourmedian[x] == -1: contourmedian[x] = y*2e4/RES
                if cumulative_distr[x, y] > 0.84  and uppersd[x] == -1: uppersd[x] = y*2e4/RES
                y += 1
            level -= 1
        twouppersd[x] = y*2e4/RES
        """
        for y in range(RES):
            cumulative += results['distr grid density'][x, y]
            for interval in contourintervals:
                if cumulative >= 0.5-interval/2 and cumulative <= 0.5+interval/2:
                    percentile_grid[x, y] += 1
        """
    boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    percentile_grid = np.transpose(percentile_grid)
    cont_colours = plt.get_cmap(name='plasma')(np.linspace(0,1,11))
    cont_colours[0] = [1, 1, 1, 1] #white
    cont = axs[1,1].contourf(results['x'], results['y'], percentile_grid, levels = boundaries, colors=cont_colours) 
    cbar = fig.colorbar(cont, ax=axs[1,1], label = 'Interval', ticks = boundaries)
    cbar.ax.set_yticklabels(['100%', '95%', '90%','80%','70%','60%','50%','40%','30%','20%','10%','0%'])
    axs[1,1].plot(twolowersd    , color = 'black', linewidth = 0.5, linestyle = (0, (1, 1)))
    axs[1,1].plot(lowersd       , color = 'black', linewidth = 0.5, linestyle = '--')
    axs[1,1].plot(contourmedian , color = 'black', linewidth = 0.5)
    axs[1,1].plot(uppersd       , color = 'black', linewidth = 0.5, linestyle = '--')
    axs[1,1].plot(twouppersd    , color = 'black', linewidth = 0.5, linestyle = (0, (1, 1)))
    axs[1,1].invert_xaxis()
    axs[1,1].set_xticks(xticklocations, xticklabels)
    axs[1,1].set_xlabel("Normalised radius $r/R$")
    axs[1,1].locator_params(axis='y', nbins=5)
    axs[1,1].set_yticks(yticklocations, yticklabels)
    axs[1,1].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")

    axs[2,0].set_title('Distribution Development Samples')
    axs[2,1].set_title('Distribution Development Samples\nfirst derivative')
    axs[2,2].set_title('Distribution Development Samples\nfirst derivative (normalised)')
    #axs[2,2].set_title('Distribution Development Samples\nsecond derivative')
    n=13*4
    x = np.linspace(-3, 3, n)
    smoothing_kernel = np.exp(-0.5*x*x)
    smoothing_kernel /= np.sum(smoothing_kernel)
    #smoothing_kernel = np.array([1])
    #print(smoothing_kernel)
    #smoothing_kernel = np.array([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])/45
    nsamples = 15
    samplearray = np.linspace(100, 1023, nsamples, dtype = np.int32)
    samples = results['distr grid density'][samplearray, :]
    dev_colours = plt.get_cmap(name='plasma')(np.linspace(0,1,nsamples))
    i=0
    for sample in samples:
        averaged_sample = np.convolve(sample, smoothing_kernel, mode = 'same')
        averaged_first_derivative = np.convolve(np.diff(averaged_sample), smoothing_kernel, mode = 'same')
        averaged_second_derivative = np.convolve(np.diff(averaged_first_derivative), smoothing_kernel, mode = 'same')
        #normalise to 1
        averaged_first_derivative_norm = averaged_first_derivative/np.max(np.abs(averaged_first_derivative))
        #averaged_second_derivative /= np.max(np.abs(averaged_second_derivative))
        axs[2,0].semilogy(np.linspace(0, 2e4, RES), averaged_sample, color = dev_colours[i])
        axs[2,1].plot(np.linspace(0, 2e4, RES-1), averaged_first_derivative, color = dev_colours[i])
        axs[2,2].plot(np.linspace(0, 2e4, RES-1), averaged_first_derivative_norm, color = dev_colours[i])
        #axs[2,2].plot(np.linspace(0, 2e4, RES-2), averaged_second_derivative, color = dev_colours[i])
        i+=1
    
    axs[2,0].set_ylim(bottom = 1e-4)
    axs[2,0].set_xlim(right = 8000)
    axs[2,1].set_yscale('symlog')
    axs[2,1].yaxis.set_major_formatter('{x:.0e}')
    axs[2,1].set_ylim(bottom = -3e-3, top = 3e-3)
    axs[2,1].set_xlim(right = 8000)
    axs[2,2].set_yscale('symlog')
    axs[2,2].yaxis.set_major_formatter('{x:.0e}')
    #axs[2,2].set_ylim(bottom = -3e-4, top = 3e-4)
    axs[2,2].set_xlim(right = 8000)

    #thiccness

    axs[2,3].set_title('Confidence Interval size')
    onesigmathiccness = uppersd - lowersd
    twosigmathiccness = twouppersd - twolowersd
    #axs[2,3].plot(onesigmathiccness, color = 'black', linewidth = 1)
    #axs[2,3].plot(twosigmathiccness, color = 'black', linewidth = 1)
    axs[2,3].fill_between(np.arange(N), 0                , onesigmathiccness, color = dev_colours[8])
    axs[2,3].fill_between(np.arange(N), onesigmathiccness, twosigmathiccness, color = dev_colours[1])
    axs[2,3].invert_xaxis()
    axs[2,3].set_xticks(xticklocations, xticklabels)
    axs[2,3].set_xlabel("Normalised radius $r/R$")
    axs[2,3].set_ylabel(r"$\Delta\rho$ [kg m$^{-3}$]")

    #minmax dens & changes
    #this is 110% sbeve certified mcterrible™

    axs[3, 0].set_title('Min Density of J explained\nmode (p), median (o), average (r)')
    mindensbinvalues, mindensbinedges, _  = axs[3, 0].hist(results['min dens'][results['dens Js explained view']], bins = bincount)
    mindensbinlocations = (mindensbinedges[:-1]+mindensbinedges[1:])/2
    modeindex = np.argmax(mindensbinvalues)
    modelocation = mindensbinlocations[modeindex]
    modevalue = mindensbinvalues[modeindex]
    weighted_indices = mindensbinvalues*np.arange(len(mindensbinvalues))
    avgindex = int(np.sum(weighted_indices)/np.sum(mindensbinvalues))
    avglocation = mindensbinlocations[avgindex]
    avgvalue = mindensbinvalues[avgindex]
    median = np.median(results['min dens'][results['dens Js explained view']])
    medianindex = np.digitize(median, mindensbinedges)
    medianlocation = mindensbinlocations[medianindex]
    medianvalue = mindensbinvalues[medianindex]

    axs[3,0].vlines(modelocation, ymin = 0, ymax = modevalue, linestyle ='--', color = 'tab:purple')
    axs[3,0].text(modelocation, modevalue, '{:.0f}'.format(modelocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', backgroundcolor='white', fontsize = 'small')
    axs[3,0].vlines(avglocation, ymin = 0, ymax = avgvalue, linestyle ='--', color = 'tab:red')
    axs[3,0].text(avglocation, avgvalue, '{:.0f}'.format(avglocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', backgroundcolor='white', fontsize = 'small')
    axs[3,0].vlines(medianlocation, ymin = 0, ymax = medianvalue, linestyle ='--', color = 'tab:orange')
    axs[3,0].text(medianlocation, medianvalue, '{:.0f}'.format(medianlocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', backgroundcolor='white', fontsize = 'small')



    axs[3, 1].set_title('Max Density of J explained\nmode (p), median (o), average (r)')
    maxdensbinvalues, maxdensbinedges, _  = axs[3, 1].hist(results['max dens'][results['dens Js explained view']], bins = bincount)
    maxdensbinlocations = (maxdensbinedges[:-1]+maxdensbinedges[1:])/2
    modeindex = np.argmax(maxdensbinvalues)
    modelocation = maxdensbinlocations[modeindex]
    modevalue = maxdensbinvalues[modeindex]
    weighted_indices = maxdensbinvalues*np.arange(len(maxdensbinvalues))
    avgindex = int(np.sum(weighted_indices)/np.sum(maxdensbinvalues))
    avglocation = maxdensbinlocations[avgindex]
    avgvalue = maxdensbinvalues[avgindex]
    median = np.median(results['max dens'][results['dens Js explained view']])
    medianindex = np.digitize(median, maxdensbinedges)
    medianlocation = maxdensbinlocations[medianindex]
    medianvalue = maxdensbinvalues[medianindex]

    axs[3,1].vlines(modelocation, ymin = 0, ymax = modevalue, linestyle ='--', color = 'tab:purple')
    axs[3,1].text(modelocation, modevalue, '{:.0f}'.format(modelocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', backgroundcolor='white', fontsize = 'small')
    axs[3,1].vlines(avglocation, ymin = 0, ymax = avgvalue, linestyle ='--', color = 'tab:red')
    axs[3,1].text(avglocation, avgvalue, '{:.0f}'.format(avglocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', backgroundcolor='white', fontsize = 'small')
    axs[3,1].vlines(medianlocation, ymin = 0, ymax = medianvalue, linestyle ='--', color = 'tab:orange')
    axs[3,1].text(medianlocation, medianvalue, '{:.0f}'.format(medianlocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', backgroundcolor='white', fontsize = 'small')

    axs[3,2].set_title('Average: start (gr) resp rhoMAX (p)\nsuccessful start (r) result (b)')
    axs[3,2].plot(results['avg start'], linestyle ='--', color = 'tab:gray')
    axs[3,2].plot(results['avg respected start'], color = 'tab:purple')
    axs[3,2].plot(results['avg successful start'], color = 'tab:red')
    axs[3,2].plot(results['avg successful result'], color = 'tab:blue')
    axs[3,2].invert_xaxis()
    axs[3,2].set_xticks(xticklocations, xticklabels)
    axs[3,2].set_xlabel("Normalised radius $r/R$")
    axs[3,2].set_yticks(yticklocations, yticklabels)
    axs[3,2].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")


    axs[3,3].set_title('Avg change')
    posneg = ['tab:blue' if y >= 0 else 'tab:red' for y in results['avg change']]
    axs[3,3].bar(range(1024), results['avg change'], width=1, bottom = 0, color = posneg)
    axs[3,3].axhline(0, linestyle ='--', color = 'tab:gray')
    axs[3,3].invert_xaxis()
    axs[3,3].set_xticks(xticklocations, xticklabels)
    axs[3,3].set_xlabel("Normalised radius $r/R$")
    axs[3,3].set_ylabel(r"$\Delta\rho$ [kg m$^{-3}$]")

    """
    #3d plot
    x = np.arange(N)
    y = np.linspace(0, 2e4, RES)
    X, Y = np.meshgrid(x, y)
    Z = np.transpose(results['distr grid density'])
    threedeefig, threedeeax = plt.subplots(subplot_kw = dict(projection = '3d'))
    threedeeax.set_title('3d Distribution')
    surf = threedeeax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='plasma', linewidth=0, edgecolor='none')
    """
    print()
    print("Number of datasets analysed:     " + str(results['n']))
    print("Number of which respect rho_MAX: " + str(results['nR']))
    print("Number of which explain Js:      " + str(results['nJ']))
    print("Percentage respect rho_MAX:      " + '{:.2%}'.format(results['rho_MAX respected%']))
    print("Percentage explained Js:         " + '{:.2%}'.format(results['Js explained%']))
    print("Time to generate plots:          " + time.strftime("%M:%S", time.gmtime((time.perf_counter()-toc))))
    plt.show()
    return

generate_plots()

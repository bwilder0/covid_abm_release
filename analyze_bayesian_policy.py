import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
frac = 1.0
country = 'lombardy'
if country == 'nyc':
    n = 8.4e6
else:
    n = 10e6
to_plot = 'infections'
#to_plot = 'infections'
names = {'nyc': 'New York',
         'hubei': 'Hubei',
         'lombardy' : 'Lombardy'}
#to_plot = 'deaths'
#distance = ''
#distance = '_distance'
styles = ['-', '--']
for distance in ['', '_distance']:
    fig, ax = plt.subplots()
    cmap = matplotlib.cm.get_cmap('viridis')
    points = [0.2, 0.5, 0.85, 0.99]
    colors = [cmap(x) for x in points]
    colors[0] = 'tab:purple'
    colors[1] = '#006992'
    colors[2] = '#06A77D'
    colors[3] = '#EDAE49'
    #D1495B
    for index in range(4):
        for frac_idx, frac in enumerate([0.5]):
    #index = 0
            combine_dir = '{}_bayesian_policy_{}_{}{}_combined'.format(country, frac, index, distance)
            param_dir = '../../parameter_sweeps/{}/'.format(combine_dir) 
        #    print((param_dir + '{}_bayesian_policy_0.5_{}{}_combined_bayesian_n{}_i0_deaths.hdf'.format(country, index, distance, n)))
            deaths = pd.read_hdf(param_dir + '{}_bayesian_policy_{}_{}{}_combined_bayesian_n{}_i0_deaths.hdf'.format(country, frac, index, distance, n)).to_numpy()
            susceptible = pd.read_hdf(param_dir + '{}_bayesian_policy_{}_{}{}_combined_bayesian_n{}_i0_susceptible.hdf'.format(country, frac, index, distance, n)).to_numpy()
            if to_plot == 'deaths':
                plt.plot(np.median(deaths, axis=0), c=colors[index], lw=3, ls = styles[frac_idx])
            if to_plot == 'infections':
                plt.plot((n - np.median(susceptible, axis=0)[10:])/n, c=colors[index], lw=3, ls = styles[frac_idx])
            end_t = susceptible.shape[1]
    
    combine_dir = '{}_bayesian_policy_baseline_combined'.format(country)
    param_dir = '../../parameter_sweeps/{}/'.format(combine_dir) 
    deaths = pd.read_hdf(param_dir + '{}_bayesian_policy_baseline_combined_bayesian_n{}_i0_deaths.hdf'.format(country, n)).to_numpy()
    susceptible = pd.read_hdf(param_dir + '{}_bayesian_policy_baseline_combined_bayesian_n{}_i0_susceptible.hdf'.format(country, n)).to_numpy()
    if to_plot == 'deaths':
        plt.plot(np.median(deaths, axis=0), c='gray', lw=3, ls = '--')
    if to_plot == 'infections':
        plt.plot((n - np.median(susceptible, axis=0)[10:])/n, c='gray', lw=3, ls = '--')

    #        print((n - np.median(susceptible, axis=0)[1:])/n)
    #    print(np.median(deaths, axis=0))
        
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    if to_plot == 'deaths':
        plt.ylabel('Total deaths', fontsize=23)
        plt.ylim(0, 200000)
    if to_plot == 'infections':
        plt.ylabel('Fraction infected', fontsize=23)
        plt.ylim(0,1)
        
    if country == 'nyc':
        if to_plot == 'infections':
            loc = 'lower right'
        else:
            loc = 'upper right'
    elif country == 'lombardy':
        if to_plot == 'deaths':
            if distance == '':
                loc = 'lower right'
            else:
                loc = 'upper right'
        else:
            loc = 'lower right'
    elif country == 'hubei':
        if to_plot == 'infections':
            if distance == '':
                loc = 'lower right'
            else:
                loc = 'upper right'
        else:
            loc = 'upper right'
    if country == 'nyc':
        leg = ax.legend(['0-19', '20-40', '40-60', '60+', 'No intervention'], fontsize=20, bbox_to_anchor=(1.06,1.075))
#    ax.legend(['0-19', '20-40', '40-60', '60+', 'No intervention'],bbox_to_anchor=(1.1, 1.05))


    plt.xlim(0, end_t-10)
    plt.xlabel('Days since $t_0$', fontsize=23)
    if country == 'hubei':
        xpos = 0.2
    else:
        xpos = 0.3
    plt.text(xpos, 0.9, '\\textbf{' + names[country] + '}', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=20)

#    plt.tight_layout()
    if country != 'nyc':
        plt.savefig('{}_{}_{}{}.pdf'.format(country, to_plot, 0.5, distance), bbox_inches='tight')
    else:
        plt.savefig('{}_{}_{}{}.pdf'.format(country, to_plot, 0.5, distance), bbox_extra_artists=(leg,), bbox_inches='tight')
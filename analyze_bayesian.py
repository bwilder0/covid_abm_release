import numpy as np 
import pandas as pd
import datetime
import json
import matplotlib.pyplot as plt
from numba import jit
import matplotlib as mpl
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

input_dict = json.load(open('inputs/hubei_bayesian_3.json', 'r'))
combine_dir = input_dict['combined_dir']
param_dir = '../../parameter_sweeps/{}/'.format(combine_dir)
if input_dict['country'] != 'NYC':
    N = 10e6
else:
    N = 8.4e6
deaths_sim = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_deaths.hdf'.format(combine_dir, N)).to_numpy()
ifr = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_ifr_tot.hdf'.format(combine_dir, N)).to_numpy().flatten()
pigc = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_pigc_vals_store.hdf'.format(combine_dir, N)).to_numpy().flatten()
dmult = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_mm_vals_store.hdf'.format(combine_dir, N)).to_numpy().flatten()
r0 = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_r0_tot.hdf'.format(combine_dir, N)).to_numpy().flatten()
sd = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_sd_vals_store.hdf'.format(combine_dir, N)).to_numpy().flatten()
susceptible = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_susceptible.hdf'.format(combine_dir, N)).to_numpy()
frac_older = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_frac_death_older.hdf'.format(combine_dir, N)).to_numpy().flatten()
critical = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_critical.hdf'.format(combine_dir, N)).to_numpy()
mild = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_mild.hdf'.format(combine_dir, N)).to_numpy()
severe = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_severe.hdf'.format(combine_dir, N)).to_numpy()
exposed = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_exposed.hdf'.format(combine_dir, N)).to_numpy()
country = input_dict['country']

if country == 'NYC':
    startinf = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_infected_start_store.hdf'.format(combine_dir, N)).to_numpy().flatten()
else:
    startinf = np.zeros(r0.shape)
    startinf[:] = 5.
italy_deaths_scale = 1


if country == 'China':
    confirmed = 68128.
elif country == 'Italy':
    confirmed = 75134
elif country == 'NYC':
    confirmed = 167478
doc = confirmed/(N - susceptible[:, -1])

implemented_state_list = []
if country == 'Italy':
    data = pd.read_csv('validation_data/italy/lombardy_data_deaths.csv')
elif country == 'China':
    data = pd.read_csv('validation_data/china/hubei.csv')
elif country == 'NYC':
    data = pd.read_csv('validation_data/US/NYC/NYC_data_deaths.csv')
elif country in implemented_state_list:
    data = pd.read_csv('validation_data/US/{}/{}_data_deaths.csv'.format(country,country))


dates = []
deaths = []
confirmed= [ ]
for i in range(len(data)):
    if country == 'Italy':
        dates.append(datetime.datetime.strptime(data['Date'][i], '%m/%d/%y %H:%M').date())
    if country == 'China':
        dates.append(datetime.datetime.strptime(data['Date'][i], '%m/%d/%y').date())
    if country == 'Republic of Korea' or country == 'US' or country in implemented_state_list or country == 'NYC':
        dates.append(datetime.datetime.strptime(data['Date'][i], '%m/%d/%Y').date())
    deaths.append(data['Deaths'][i])
    if country == 'Republic of Korea':
        confirmed.append(data['Confirmed'][i])
        
deaths = np.array(deaths).astype(np.float)

if country == 'NYC':
    deaths = np.cumsum(deaths)
    
if country == 'China':
    final_total = deaths[-1]
    prev_total = deaths[85]
    deaths[85:] = prev_total
    deaths *= final_total/prev_total
if country == 'Italy':
    deaths *= italy_deaths_scale
if country == 'NYC':
    deaths *= 1

if 'start_date_list' in input_dict:
    d0  = datetime.datetime.strptime(input_dict['start_date_list'][0], '%m-%d-%Y').date()
else:
     d0  = datetime.datetime.strptime(input_dict['start_date_range'][0], '%m-%d-%Y').date()

d_end = datetime.datetime.strptime(input_dict['d_end'], '%m-%d-%Y').date()
d_lockdown = datetime.datetime.strptime(input_dict['d_lockdown'], '%m-%d-%Y').date()


if dates[0] > d0:
    num_to_add = (dates[0] - d0).days
    deaths = [0]*num_to_add + list(deaths)
    deaths = np.array(deaths)
    dates = [d0 + datetime.timedelta(days=i) for i in range(num_to_add)] + dates
elif dates[0] < d0:
    idx = dates.index(d0)
    deaths = deaths[idx:]
    dates = dates[idx:]
if dates[-1] > d_end:
    idx = dates.index(d_end)
    dates = dates[:idx+1]
    deaths = deaths[:idx+1]
elif dates[-1] > d_end:
    raise Exception('Incomplete validation data')

accept = np.abs(deaths_sim[:, -1] - deaths[-1]) < 100

from scipy.special import gammaln
def nb_likelihood(preds, observations, phi):
    return gammaln(phi + observations) - gammaln(observations + 1) - gammaln(phi) + phi*np.log(phi/(phi + preds)) + observations*np.log(preds/(phi + preds))
@jit
def sample_posterior(log_posterior, n_samples):
    samples = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        gum = np.random.gumbel(size=log_posterior.shape[0])
        samples[i] = np.argmax(log_posterior + gum)
    return samples
   

def smooth_zeros(new_deaths):
    new_deaths = np.copy(new_deaths)
    zeros = np.where(new_deaths == 0)[0]
    first_nonzero = np.where(new_deaths != 0)[0][0]
    for i in zeros:
        if i > first_nonzero and i < new_deaths.shape[0] - 1:
            total = new_deaths[i] + new_deaths[i+1]
            new_deaths[i] = new_deaths[i+1] = total/2
    return new_deaths
    
    
new_deaths = deaths_sim[:, 1:] - deaths_sim[:, :-1]
new_deaths_obs = deaths[1:]-deaths[:-1]
new_deaths_obs = smooth_zeros(new_deaths_obs)
if country == 'China':
    sigma_nb = 0.4598
elif country == 'Italy':
    sigma_nb = 0.2591
elif country == 'NYC':
    sigma_nb = 0.0641

include_lower = np.where(new_deaths_obs >= 20)[0][0]
include_upper =  np.where(new_deaths_obs >= 20)[0][-1]     
lik = nb_likelihood(new_deaths[:, include_lower:include_upper].flatten() + 0.1, new_deaths_obs[include_lower:include_upper].reshape(1, -1).repeat(new_deaths.shape[0], axis=0).flatten(), 1/sigma_nb).reshape(new_deaths[:, include_lower:include_upper].shape).sum(axis=1) 


log_prior = np.log(np.ones(lik.shape[0])/lik.shape[0])
log_posterior = lik + log_prior

posterior_samples = sample_posterior(log_posterior, 10000)

if country == 'China':
    ifr_range = [0.004, 0.008]
    posterior_samples = posterior_samples[(ifr[posterior_samples] > ifr_range[0])*(ifr[posterior_samples] < ifr_range[1])]

def ci(a, alpha):
    a = a.copy()
    a.sort()
    return a[int((alpha/2)*a.shape[0])], a[int((1-(alpha/2))*a.shape[0])]


accept = posterior_samples
t_lockdown = (d_lockdown - d0).days

time_from_d0 = []
for i in range(len(dates)):
    time_from_d0.append((dates[i] - d0).days)

plt.figure()
for i in range(min((deaths_sim[accept].shape[0], 200))):
    plt.plot(deaths_sim[accept][i], alpha = 0.06, c='C0')
if country == 'Italy' or country == 'NYC' or country in implemented_state_list:
    plt.scatter(time_from_d0, deaths, color='k', s=10)
elif country == 'China':
    plt.scatter(time_from_d0[::2], deaths[::2], color='k', s=10)
plt.plot(np.median(deaths_sim[accept], axis=0), color = 'g', lw = 2.5)

plt.ylabel('Total deaths', fontsize=23)
if country == 'Italy':
    plt.ylim(0, 30000)
plt.vlines(t_lockdown, 0, plt.ylim()[1], linestyles = '--', color='r')

date_labels = []
for i in range(deaths_sim.shape[1]):
    new_date = d0 + datetime.timedelta(days=i)
    date_labels.append('{}/{}'.format(new_date.month, new_date.day))

plt.tick_params(axis='both', which='major', labelsize=20)
if country ==  'China':
    currticks = [0, 30, 60, 90, 120, 150, 171]
elif country == 'Italy':
    currticks = [0, 20, 40, 60, 80, 104]
elif country == 'NYC':
    currticks = [0, 15, 30, 45, 60, 75]
elif country == 'US':
    currticks = [0, 10, 20, 30, 40, 50, 60, 70, 80]
elif country in implemented_state_list:
    currticks = [0, 10, 20, 30, 40]
plt.xticks(currticks)
plt.xticks(currticks, [date_labels[i] for i in currticks], rotation=45)
plt.xlabel('Date', fontsize=23)
plt.xlim(0, deaths_sim.shape[1]-1)
plt.tight_layout()
plt.savefig('img/totaldeaths_{}.pdf'.format(country))



fig, ax = plt.subplots()
if country == 'Italy' or country == 'NYC' or country in implemented_state_list or country == 'China':
    plt.scatter(time_from_d0[1:], new_deaths_obs, color='k', s=10)

for i in range(min((new_deaths.shape[0], 50))):
    plt.plot(new_deaths[accept][i], alpha = 0.06, c='C0')

plt.plot(np.median(new_deaths[accept], axis=0), color = 'g', lw = 2.5)
plt.ylabel('Daily new deaths', fontsize=23)
plt.vlines(t_lockdown, 0, 1.2*new_deaths.max(), linestyles = '--', color='r')

date_labels = []
for i in range(deaths_sim.shape[1]):
    new_date = d0 + datetime.timedelta(days=i)
    date_labels.append('{}/{}'.format(new_date.month, new_date.day))

plt.tick_params(axis='both', which='major', labelsize=20)
if country ==  'China':
    currticks = [0, 30, 60, 90, 120, 150, 171]
elif country == 'Italy':
    currticks = [0, 20, 40, 60, 80, 104]
elif country == 'US':
    currticks = [0, 10, 20, 30, 40, 50, 60, 70, 80]
elif country == 'NYC':
    currticks = [0, 15, 30, 45, 60, 75]
elif country in implemented_state_list:
    currticks = [0, 10, 20, 30, 40]
plt.xticks(currticks)
plt.xticks(currticks, [date_labels[i] for i in currticks], rotation=45)
plt.xlabel('Date', fontsize=23)
plt.xlim(0, deaths_sim.shape[1]-1)
plt.tight_layout()
if country == 'Italy':
    plt.ylim(0, 750)
elif country == 'China':
    plt.ylim(0, 750)
elif country == 'NYC':
    plt.ylim(0, 750)
    
names = {'NYC': 'New York',
         'China': 'Hubei',
         'Italy' : 'Lombardy'}
    
if country == 'China':
    xpos = 0.2
else:
    xpos = 0.3
plt.text(xpos, 0.9, '\\textbf{' + names[country] + '}', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=20)

plt.savefig('img/newdeaths_{}.pdf'.format(country))


posterior_save = {}
posterior_save['p_infect_given_contact_list'] = pigc[posterior_samples]
posterior_save['mortality_multiplier_list'] = dmult[posterior_samples]
posterior_save['start_infected_list'] = startinf[posterior_samples]
posterior_save['start_date_list'] = sd[posterior_samples]
import pickle
pickle.dump(posterior_save, open('{}_posterior.pickle'.format(country), 'wb'))

new_deaths = deaths_sim[:, 1:] - deaths_sim[:, :-1]
new_deaths_obs = deaths[1:]-deaths[:-1]
new_deaths_obs = smooth_zeros(new_deaths_obs)

if country == 'Italy':
    cutoff_idxs = [55, 65, 75, 85, 95]
elif country == 'China':
    cutoff_idxs = [80, 90, 100, np.where(new_deaths_obs >= 20)[0][-1] ]
if country == 'NYC':
    cutoff_idxs = [45, 50, 65, 75, 85]
    
    
def sample_posterior_observations(posterior_deaths, sigma_nb):
    r = 1./sigma_nb
    p = posterior_deaths/(posterior_deaths + r)
    return np.random.negative_binomial(r, 1- p.flatten()).reshape(p.shape)

def posterior_observations_ci(posterior_deaths, sigma_nb, alpha):
    obs_samples = sample_posterior_observations(posterior_deaths, sigma_nb)
    upper_ci = np.zeros(posterior_deaths.shape[1])
    lower_ci = np.zeros(posterior_deaths.shape[1])
    for t in range(upper_ci.shape[0]):
        a, b = ci(obs_samples[:, t], alpha)
        upper_ci[t] = b
        lower_ci[t] = a
    return lower_ci, upper_ci


#

#
if country == 'China':
    sigma_nb = 0.4598
elif country == 'Italy':
    sigma_nb = 0.2591

include_lower = np.where(new_deaths_obs >= 20)[0][0]
include_upper =  np.where(new_deaths_obs >= 20)[0][-1] 


for include_upper in cutoff_idxs:
    lik = nb_likelihood(new_deaths[:, include_lower:include_upper].flatten() + 0.1, new_deaths_obs[include_lower:include_upper].reshape(1, -1).repeat(new_deaths.shape[0], axis=0).flatten(), 1/sigma_nb).reshape(new_deaths[:, include_lower:include_upper].shape).sum(axis=1) 
    
    log_prior = np.log(np.ones(lik.shape[0])/lik.shape[0])
    log_posterior = lik + log_prior
    
    posterior_samples_new = sample_posterior(log_posterior, 10000)
    if country == 'China':
        ifr_range = [0.004, 0.008]
        posterior_samples_new = posterior_samples_new[(ifr[posterior_samples_new] > ifr_range[0])*(ifr[posterior_samples_new] < ifr_range[1])]

    accept = posterior_samples_new
    
    def sample_posterior_observations(posterior_deaths, sigma_nb):
        r = 1./sigma_nb
        p = posterior_deaths/(posterior_deaths + r)
        return np.random.negative_binomial(r, 1- p.flatten()).reshape(p.shape)
    
    def posterior_observations_ci(posterior_deaths, sigma_nb, alpha):
        obs_samples = sample_posterior_observations(posterior_deaths, sigma_nb)
        upper_ci = np.zeros(posterior_deaths.shape[1])
        lower_ci = np.zeros(posterior_deaths.shape[1])
        for t in range(upper_ci.shape[0]):
            a, b = ci(obs_samples[:, t], alpha)
            upper_ci[t] = b
            lower_ci[t] = a
        return lower_ci, upper_ci
    
    lower_ci, upper_ci = posterior_observations_ci(new_deaths[accept], sigma_nb, 0.1)
    
    plt.figure()
    plt.scatter(time_from_d0[1:include_upper+1], new_deaths_obs[:include_upper], color='mediumorchid', s=10)
    plt.scatter(time_from_d0[include_upper+1:], new_deaths_obs[include_upper:], color='k', s=10)
    
    plt.plot(np.arange(1, new_deaths.shape[1]+1), np.median(new_deaths[accept], axis=0), color = 'g', lw = 2.5)
    plt.fill_between(np.arange(include_upper+1, new_deaths.shape[1]+1), lower_ci[include_upper:], upper_ci[include_upper:], alpha=0.3)
    plt.ylabel('Daily new deaths', fontsize=23)
    plt.vlines(include_upper+1, 0, 1.2*new_deaths.max(), linestyles = '--', color='k')
    
    
    date_labels = []
    for i in range(deaths_sim.shape[1]):
        new_date = d0 + datetime.timedelta(days=i)
        date_labels.append('{}/{}'.format(new_date.month, new_date.day))
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    #currticks = plt.xticks()
    if country ==  'China':
        currticks = [0, 30, 60, 90, 120, 150, 171]
    elif country == 'Italy':
        currticks = [0, 20, 40, 60, 80, 104]
    elif country == 'US':
        currticks = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    elif country == 'NYC':
        currticks = [0, 15, 30, 45, 60, 75]
    elif country in implemented_state_list:
        currticks = [0, 10, 20, 30, 40]
    plt.xticks(currticks)
    plt.xticks(currticks, [date_labels[i] for i in currticks], rotation=45)
    plt.xlabel('Date', fontsize=23)
    plt.xlim(0, deaths_sim.shape[1]-1)
    plt.tight_layout()
    if country == 'Italy':
        plt.ylim(0, 1500)
    elif country == 'China':
        plt.ylim(0, 500)
    elif country == 'NYC':
        plt.ylim(0, 1000)

    plt.savefig('img/{}_predposterior_{}.pdf'.format(country, include_upper))

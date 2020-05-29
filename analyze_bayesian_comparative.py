import numpy as np 
import pandas as pd
import datetime
import json
import matplotlib.pyplot as plt
from numba import jit

files = ['inputs/lombardy_bayesian.json', 'inputs/hubei_bayesian.json', 'inputs/nyc_bayesian.json']

all_ifr = {}
all_r0 = {}
all_dmult = {}
all_doc = {}
all_susceptible = {}

italy_deaths_scale = 1
for file in files:
    
    input_dict = json.load(open(file, 'r'))
    combine_dir = input_dict['combined_dir']
    param_dir = '../../parameter_sweeps/{}/'.format(combine_dir)
    country = input_dict['country']
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
#    critical = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_critical.hdf'.format(combine_dir, N)).to_numpy()
#    mild = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_mild.hdf'.format(combine_dir, N)).to_numpy()
#    severe = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_severe.hdf'.format(combine_dir, N)).to_numpy()
#    exposed = pd.read_hdf(param_dir + '{}_bayesian_n{}_i0_exposed.hdf'.format(combine_dir, N)).to_numpy()
    
    
    sd = np.array([int(sd[i][3:5]) for i in range(len(sd))])
    
    
    
    country = input_dict['country']
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
        deaths *= italy_deaths_scale
    
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

    include_lower = np.where(new_deaths_obs >= 20)[0][0]
    include_upper =  np.where(new_deaths_obs >= 20)[0][-1]     
    lik = nb_likelihood(new_deaths[:, include_lower:include_upper].flatten() + 0.1, new_deaths_obs[include_lower:include_upper].reshape(1, -1).repeat(new_deaths.shape[0], axis=0).flatten(), 1/sigma_nb).reshape(new_deaths[:, include_lower:include_upper].shape).sum(axis=1) 
    
       
    
    log_prior = np.log(np.ones(lik.shape[0])/lik.shape[0])
    log_posterior = lik + log_prior
    
        
    #gum = np.random.gumbel(size=(10000 , all_log_posterior.shape[0]))
    posterior_samples = sample_posterior(log_posterior, 10000)
    
    if country == 'China':
        ifr_range = [0.004, 0.008]
        posterior_samples = posterior_samples[(ifr[posterior_samples] > ifr_range[0])*(ifr[posterior_samples] < ifr_range[1])]

    print(len(np.unique(posterior_samples)))
    print(np.median((N - susceptible[posterior_samples, -1])/N))
    all_ifr[country] = ifr[posterior_samples]
    all_r0[country] = r0[posterior_samples]
    all_doc[country] = doc[posterior_samples]
    all_dmult[country] = dmult[posterior_samples]

plt.figure()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.hist(all_doc['China'], bins=10, density=True, alpha=0.3)
plt.hist(all_doc['Italy'], bins=10, density=True, alpha=0.3)
plt.hist(all_doc['NYC'], bins=10, density=True, alpha=0.3)

plt.legend(['Hubei', 'Lombardy', 'NYC'], fontsize=20)
plt.xlabel('Documentation rate', fontsize=23)
plt.ylabel('Posterior density', fontsize=23)
plt.tight_layout()
plt.savefig('img/doc_compare_{}.pdf'.format(italy_deaths_scale))

plt.figure()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.hist(all_r0['China'], bins=10, density=True, alpha=0.3)
plt.hist(all_r0['Italy'], bins=10, density=True, alpha=0.3)
plt.hist(all_r0['NYC'], bins=10, density=True, alpha=0.3)

plt.legend(['Hubei', 'Lombardy', 'NYC'], fontsize=20)
plt.xlabel('$r_0$', fontsize=25)
plt.ylabel('Posterior density', fontsize=23)
plt.xlim(1.75, 4)
plt.tight_layout()
plt.savefig('img/r0_compare_{}.pdf'.format(italy_deaths_scale))




def ci(a, alpha):
    a = a.copy()
    a.sort()
    return a[int((alpha/2)*a.shape[0])], a[int((1-(alpha/2))*a.shape[0])]

for country in all_r0:
    print(country, np.median(all_r0[country]), ci(all_r0[country], 0.1))
    
for country in all_doc:
    print(country, np.median(all_doc[country]), ci(all_doc[country], 0.1))
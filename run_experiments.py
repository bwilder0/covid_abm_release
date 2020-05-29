import numpy as np
from numba import jit
import sample_households
import global_parameters
import scipy.special
import sample_comorbidities
import csv
import functions
import pickle
from datetime import date
import numba
from seir_individual import run_model
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error
import os
import argparse
from itertools import product
import time
import json 

FILESTART = time.time()


# Test
load_population = False
N_INFECTED_START = 5.


parser = argparse.ArgumentParser(description='Run lombardy parameter sweeps in a distributed way.')
parser.add_argument('--do_the_math', action='store_true', default=False, help='Given N_SIMS_PER_COMBO,\
                N_SIMS_PER_JOB, and the parameter ranges (see source code), will report the range of\
                indexes that you should pass to the slurm job script')
parser.add_argument('--N_SIMS_PER_COMBO', type=int, default=10, help='Number of simulations you plan to run per parameter combo')
parser.add_argument('--N_SIMS_PER_JOB', type=int, default=10, help='Number of simulations to run per job. Must be divisible by N_SIMS_PER_COMBO.')
parser.add_argument('--N_PARAMS_PER_JOB', type=int, default=1, help='Number of simulations to run per job. Must be divisible by N_SIMS_PER_COMBO.')

parser.add_argument('--index', type=int, default=0, help='Index of the job to run. --see do_the_math')
parser.add_argument('--seed_offset', type=int, default=0, help='seed = index + index_offset. \
    So to run a new indepent set of simulations for the same parameter combos, this \
    value must be larger than the total number of simulations you have run so far, \
    i.e., total_combos * N_SIMS_PER_COMBO.')
parser.add_argument('--sim_name', type=str, default="whole_US_0", help='Always specify a new directory for your jobs!')
parser.add_argument('-t','--numthreads', type=int, default=4, help='Number of threads to use during computation.')
parser.add_argument('-N','--popsize', type=float, default=1e7, help='Population size to run')
parser.add_argument('-l','--lockdown', type=float, default=2, help='lockdown factor')
parser.add_argument('--json', type=str, help='JSON file of input parameters')
parser.add_argument('--popdir', type=str, default='', help='Directory to load population files from')
parser.add_argument('--EXP_ROOT_DIR', type=str, default='parameter_sweeps', help="On fasrc, you should specify a directory\
                                            that you've created in long term file storage, \
                                            e.g., /n/holylfs/LABS/tambe_lab/jkillian/covid19/parameter_sweeps.\
                                            IMPORTANT: make sure you have already created this directory (this script will assume it exists.)")

args = parser.parse_args()

EXP_ROOT_DIR = args.EXP_ROOT_DIR

if args.json is None:
    args.json = 'inputs/' + args.sim_name + '.json'

N = float(args.popsize)

print('before load')
input_dict = json.load(open(args.json, 'r'))
print('load')
be_bayesian = 'bayesian' in input_dict and input_dict['bayesian'] == 'True'
if not be_bayesian:
    p_infect_given_contact_list = input_dict['p_infect_given_contact_list']
    mortality_multiplier_list = input_dict['mortality_multiplier_list']
    start_date_list = input_dict['start_date_list']
    pinf_mult_list = input_dict['pinf_mult_list']
    master_combo_list = list(product(p_infect_given_contact_list, mortality_multiplier_list, start_date_list, pinf_mult_list))

else:
    if 'load_posterior' in input_dict and input_dict['load_posterior'] == 'True':
        print('loading posterior')
        posterior_samples = pickle.load(open(input_dict['posterior_file'], 'rb'))
        p_infect_given_contact_list = posterior_samples['p_infect_given_contact_list']
        start_date_list = posterior_samples['start_date_list']
        mortality_multiplier_list = posterior_samples['mortality_multiplier_list']
        start_infected_list = posterior_samples['start_infected_list']
        pinf_mult_list = np.zeros(start_infected_list.shape)
        pinf_mult_list[:] = input_dict['pinf_mult_range'][0]
    else:
        lhs = pd.read_hdf('lhs.hdf').to_numpy()
        p_infect_given_contact_range = input_dict['p_infect_given_contact_range']
        mortality_multiplier_range = input_dict['mortality_multiplier_range']
        if 'start_date_range' in input_dict:
            start_date_range = input_dict['start_date_range']
            start_early = datetime.datetime.strptime(start_date_range[0], '%m-%d-%Y').date()
            start_late = datetime.datetime.strptime(start_date_range[1], '%m-%d-%Y').date()
            start_date_list = []
            for i in range((start_late - start_early).days + 1):
                start_date_list.append((start_early + datetime.timedelta(days=i)).strftime('%m-%d-%Y'))
        else:
            start_date_list = input_dict['start_date_list']
        if 'start_infected_range' in input_dict:
            start_infected_range = input_dict['start_infected_range']
            start_infected_list = np.floor(lhs[:, 3]*(start_infected_range[1] - start_infected_range[0])) + start_infected_range[0]
        else:
            start_infected_list = np.zeros(lhs.shape[0])
            start_infected_list[:] = N_INFECTED_START
        pinf_mult_range = input_dict['pinf_mult_range']
        if len(p_infect_given_contact_range) > 1:
            p_infect_given_contact_list = lhs[:, 0] * (p_infect_given_contact_range[1] - p_infect_given_contact_range[0]) + p_infect_given_contact_range[0]
        else:
            p_infect_given_contact_list = np.array(p_infect_given_contact_range*lhs.shape[0])
        if len(mortality_multiplier_range) > 1:
            mortality_multiplier_list = lhs[:, 1] * (mortality_multiplier_range[1] - mortality_multiplier_range[0]) + mortality_multiplier_range[0]
        else:
            mortality_multiplier_list = np.array(mortality_multiplier_range*lhs.shape[0])
        if len(pinf_mult_range) > 1:
            pinf_mult_list = lhs[:, 2] * (pinf_mult_range[1] - pinf_mult_range[0]) + pinf_mult_range[0]
        else:
            pinf_mult_list = np.array(pinf_mult_range*lhs.shape[0])
        start_date_idx = np.floor(lhs[:, 3]*len(start_date_list)).astype(np.int)
        start_date_list = np.array(start_date_list)[start_date_idx]
    
    


num_p = len(p_infect_given_contact_list)
num_m = len(mortality_multiplier_list)
num_d = len(start_date_list)
num_pim = len(pinf_mult_list)


N_SIMS_PER_COMBO = args.N_SIMS_PER_COMBO
N_SIMS_PER_JOB = args.N_SIMS_PER_JOB

total_combos = num_p*num_m*num_d*num_pim

if not be_bayesian and N_SIMS_PER_COMBO % N_SIMS_PER_JOB != 0:
    raise ValueError('Please ensure that N_SIMS_PER_COMBO is divisible by N_SIMS_PER_JOB')

print('before simpath')
sim_name = args.sim_name
sim_path = os.path.join(EXP_ROOT_DIR, sim_name)
print(sim_path)
if not os.path.exists(sim_path):
    print('making dir: ' + str(sim_path))
    try:
        os.makedirs(sim_path)
    except:
        pass
print('after make dir')
if args.do_the_math:

    n_indices = int(total_combos*N_SIMS_PER_COMBO/N_SIMS_PER_JOB)
    print()
    print("NUM COMBOS:",total_combos)
    print("NUM INDICES:",n_indices)
    print()
    print("Please launch jobs indexed from 0 to %s, e.g.,"%(n_indices-1))
    print("mkdir %s; sbatch --array=0-%s job.parameter_sweep_lombardy.sh %s %s %s %s"%(sim_path, n_indices-1, N_SIMS_PER_COMBO, N_SIMS_PER_JOB, args.sim_name, args.seed_offset))
    print()
    print("However, please note that you can only queue 10000 jobs at a time!")
    print("")
    print("Note: the larger N_SIMS_PER_JOB, the more time and memory it will take.")
    print("But the smaller N_SIMS_PER_JOB, the less time and memory it will take, but more jobs and CPUs.")
    exit()

elif args.index == -1:
    raise ValueError('Please set the index or pass --do_the_math.')


INDEX = args.index

# TODO: Pass in a parameter
SEED_OFFSET = args.seed_offset


"""Run Parameter Sweep
"""

if not be_bayesian:
    JOBS_PER_COMB0 = N_SIMS_PER_COMBO//N_SIMS_PER_JOB
    
    param_combo_index = INDEX//JOBS_PER_COMB0
    TRIAL_NUMBER = INDEX%JOBS_PER_COMB0

seed = INDEX * N_SIMS_PER_JOB + SEED_OFFSET

params = numba.typed.Dict()
params['seed'] = float(seed)
"""int: Seed for random draws"""

np.random.seed(seed)

country = input_dict['country']

d_lockdown = datetime.datetime.strptime(input_dict['d_lockdown'], '%m-%d-%Y').date()
d_stay_home_start = date(3020, 8, 8) # did any states have full stay-at-home orders?
d_end = None
try:
    d_end = datetime.datetime.strptime(input_dict['d_end'], '%m-%d-%Y').date()
except:
    d_end = date(2020, 4, 16)
try:
    d_lockdown_release  = datetime.datetime.strptime(input_dict['d_lockdown_release'], '%m-%d-%Y').date()
except KeyError:
    print('d_lockdown_release not found, setting to 2030')
    d_lockdown_release = date(2030, 1, 1) #Never release unless explicitly set

d_school_lockdown  = None
try:
    d_school_lockdown  = datetime.datetime.strptime(input_dict['d_school_lockdown'], '%m-%d-%Y').date()
except KeyError:
    print('d_school_lockdown not found, setting to 2030')
    d_school_lockdown = date(2030, 1, 1) # Make lockdown date far out; causes schools to close on d_lockdown

if be_bayesian:
    if 'start_date_list' in input_dict:
        d_earliest_start = datetime.datetime.strptime(input_dict['start_date_list'][0], '%m-%d-%Y').date()
    else:
        d_earliest_start = datetime.datetime.strptime(input_dict['start_date_range'][0], '%m-%d-%Y').date()
    T = float((d_end - d_earliest_start).days + 1)
    print(d_earliest_start)
    params['T'] = T
    T = int(T)

else:
    this_combo = master_combo_list[param_combo_index]
    sd_val = this_combo[2]
    d0  = datetime.datetime.strptime(sd_val, '%m-%d-%Y').date()
    T = float((d_end - d0).days + 1)
    params['T'] = T
    T = int(T)

print(T)
params['n'] = N

params['n_threads'] = float(args.numthreads)
n_ages = 101
"""int: Number of ages. Currently ages 0-100, with each age counted separately"""
params['n_ages'] = float(n_ages)

mean_time_to_isolate_factor = ((0, 14, 1), (14, 24, 1), (25, 39, 1), (40, 69, 1), (70, 100, 1))
mean_time_to_isolate_factor = np.array(mean_time_to_isolate_factor)


"""2c. Set transition times between states
We assume that all state transitions are exponentially distributed.
"""

#for now, the time for all of these events will be exponentially distributed
#from https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf
params['mean_time_to_severe'] = 7.
params['mean_time_mild_recovery'] = 14.


"""TODO: Find documented probabilities, age distribution or mean_time"""
cumulative_documentation_mild = 0.00000000001
params['p_documented_in_mild'] = global_parameters.calibrate_p_document_mild(cumulative_documentation_mild, country, None, params['mean_time_mild_recovery'], None)



#guessing based on time to mechanical ventilation as 14.5 days from
#https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext
#and subtracting off the 7 to get to critical. This also matches mortality risk
#starting at 2 weeks in the WHO report
params['mean_time_to_critical'] = 7.5

#WHO gives 3-6 week interval for severe and critical combined
#using 4 weeks as mean for severe and 5 weeks as mean for critical
params['mean_time_severe_recovery'] = 28. - params['mean_time_to_severe']
params['mean_time_critical_recovery'] = 35. - params['mean_time_to_severe'] - params['mean_time_to_critical'] 
#mean_time_severe_recovery = mean_time_critical_recovery = 21

#mean_time_to_death = 35 #taking the midpoint of the 2-8 week interval
#update: use 35 - mean time to severe - mean time to critical as the excess time
#to death after reaching critical
#update: use 18.5 days as median time onset to death from
#https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext
params['mean_time_to_death'] = 18.5 - params['mean_time_to_severe'] - params['mean_time_to_critical'] 
#mean_time_to_death = 1 #this will now be critical -> death

#distribution of time until symptom onset
#set based on https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported
params['time_to_activation_mean'] = 1.621
params['time_to_activation_std'] = 0.418


fraction_stay_home = np.zeros(n_ages)
fraction_stay_home[:] = 0
if 'fraction_stay_home' in input_dict:
    fraction_stay_home_groups = input_dict['fraction_stay_home']
    stay_home_groups = input_dict['stay_home_groups']
    for i in range(len(stay_home_groups)-1):
        fraction_stay_home[stay_home_groups[i]:stay_home_groups[i+1]] = fraction_stay_home_groups[i]
print(fraction_stay_home)    

params['mean_time_to_isolate_asympt'] = 10000.

params['mean_time_to_isolate'] = 4.6
"""float: Time from symptom onset to isolation
https://www.nejm.org/doi/pdf/10.1056/NEJMoa2001316?articleTools=true
optimistic estimate is time to seek first medical care, pessimistic is time to hospital admission
"""

params['asymptomatic_transmissibility'] = 0.55
"""float: How infectious are asymptomatic cases relative to symptomatic ones
https://science.sciencemag.org/content/early/2020/03/13/science.abb3221
"""

# DON'T CHANGE: we don't want p infect household to recalibrate for different policy what ifs on mean time to isolate
MEAN_TIME_TO_ISOLATE = 4.6 # DON'T CHANGE
p_infect_household = global_parameters.get_p_infect_household(4.6, params['time_to_activation_mean'], params['time_to_activation_std'], params['asymptomatic_transmissibility'])
params['p_infect_household'] = p_infect_household


overall_p_critical_death = 0.49
"""float: Probability that a critical individual dies. This does _not_ affect
overall mortality, which is set separately, but rather how many individuals
end up in critical state. 0.49 is from
http://weekly.chinacdc.cn/en/article/id/e53946e2-c6c4-41e9-9a9b-fea8db1a8f51
"""


"""2. LOAD AND CALCULATE NON-FREE PARAMETERS
"""



# households, age = sample_households.sample_households_un(n,country)

"""2a.  Construct contact matrices
Idea: based on his/her age, each individuals has a different probability
      of contacting other individuals depending on their age
Goal: construct contact_matrix, which states that an individual of age i
     contacts Poission(contact[i][j]) contacts with individuals of age j
The data we have for this is based on contacts between individuals in age
intervals and must be converted.
"""

contact_matrix_age_groups_dict = {
    'infected_1': '0-4', 'contact_1': '0-4', 'infected_2': '5-9',
    'contact_2': '5-9', 'infected_3': '10-14', 'contact_3': '10-14',
    'infected_4': '15-19', 'contact_4': '15-19', 'infected_5': '20-24',
    'contact_5': '20-24', 'infected_6': '25-29', 'contact_6': '25-29',
    'infected_7': '30-34', 'contact_7': '30-34', 'infected_8': '35-39',
    'contact_8': '35-39', 'infected_9': '40-44', 'contact_9': '40-44',
    'infected_10': '45-49', 'contact_10': '45-49', 'infected_11': '50-54',
    'contact_11': '50-54', 'infected_12': '55-59', 'contact_12': '55-59',
    'infected_13': '60-64', 'contact_13': '60-64', 'infected_14': '65-69',
    'contact_14': '65-69', 'infected_15': '70-74', 'contact_15': '70-74',
    'infected_16': '75-79', 'contact_16': '75-79'}
"""dict: Mapping from interval names to age ranges."""

def read_contact_matrix(country, setting, do_extrapolation):
    """Create a country-specific contact matrix from stored data.

    Read a stored contact matrix based on age intervals. Return a matrix of
    expected number of contacts for each pair of raw ages. Extrapolate to age
    ranges that are not covered.

    Args:
        country (str): country name.

    Returns:
        float n_ages x n_ages matrix: expected number of contacts between of a person
            of age i and age j is Poisson(matrix[i][j]).
    """
    matrix = np.zeros((n_ages, n_ages))
    with open('Contact_Matrices/{}/{}_{}.csv'.format(country, setting, country), 'r') as f:
        csvraw = list(csv.reader(f))
    col_headers = csvraw[0][1:-1]
    row_headers = [row[0] for row in csvraw[1:]]
    data = np.array([row[1:-1] for row in csvraw[1:]])
    for i in range(len(row_headers)):
        for j in range(len(col_headers)):
            interval_infected = contact_matrix_age_groups_dict[row_headers[i]]
            interval_infected = [int(x) for x in interval_infected.split('-')]
            interval_contact = contact_matrix_age_groups_dict[col_headers[j]]
            interval_contact = [int(x) for x in interval_contact.split('-')]
            for age_infected in range(interval_infected[0], interval_infected[1]+1):
                for age_contact in range(interval_contact[0], interval_contact[1]+1):
                    matrix[age_infected, age_contact] = float(data[i][j])/(interval_contact[1] - interval_contact[0] + 1)
    if do_extrapolation:
        # extrapolate from 79yo out to 100yo
        # start by fixing the age of the infected person and then assuming linear decrease
        # in their number of contacts of a given age, following the slope of the largest
        # pair of age brackets that doesn't contain a diagonal term (since those are anomalously high)
        for i in range(interval_infected[1]+1):
            if i < 65: # 0-65
                slope = (matrix[i, 70] - matrix[i, 75])/5
            elif i < 70: # 65-70
                slope = (matrix[i, 55] - matrix[i, 60])/5
            elif i < 75: # 70-75
                slope = (matrix[i, 60] - matrix[i, 65])/5
            else: # 75-80
                slope = (matrix[i, 65] - matrix[i, 70])/5
    
            start_age = 79
            if i >= 75:
                start_age = 70
            for j in range(interval_contact[1]+1, n_ages):
                matrix[i, j] = matrix[i, start_age] - slope*(j - start_age)
                if matrix[i, j] < 0:
                    matrix[i, j] = 0
    
        # fix diagonal terms
        for i in range(interval_infected[1]+1, n_ages):
            matrix[i] = matrix[interval_infected[1]]
        for i in range(int((100-80)/5)):
            age = 80 + i*5
            matrix[age:age+5, age:age+5] = matrix[79, 79]
            matrix[age:age+5, 75:80] = matrix[75, 70]
        matrix[100, 95:] = matrix[79, 79]
        matrix[95:, 100] = matrix[79, 79]

    return matrix

if country == 'China':
    other_contact = read_contact_matrix(country, 'other', True) + read_contact_matrix(country, 'work', False) + read_contact_matrix(country, 'school', False)
    school_contact = np.zeros(other_contact.shape)
    #rescale to get total contacts matching Wuhan from https://science.sciencemag.org/content/early/2020/05/04/science.abb8001
    total_contacts = np.array([8.6, 16.2, 15.3, 13.8, 13.9]) - 0.94*np.array([2.2, 2.1, 2.1, 2, 1.4])
    intervals = np.array([0, 7, 20, 40, 60, 101])
    
    contact_distr = np.diag(1/other_contact.sum(axis=1))@other_contact
    
    num_contacts = np.zeros(101)
    for i in range(len(intervals)-1):
        num_contacts[intervals[i]:intervals[i+1]] = total_contacts[i]
        
    rescaled_contacts = np.diag(num_contacts)@contact_distr
    other_contact = rescaled_contacts


else:
    other_contact = read_contact_matrix(country, 'other', True) + read_contact_matrix(country, 'work', False)
    school_contact = read_contact_matrix(country, 'school', False)
work_contact = np.zeros((1, n_ages, n_ages))
customer_contact = np.zeros((1, n_ages, n_ages))
#distribution parameters for number of work contacts -- not used in these experiments
work_params = pd.read_csv('work_contact_params.csv')
for col in work_params.columns:
    params[col] = work_params[col][0]

"""n_ages x n_ages matrix: expected number of contacts between of a person
    of age i and age j is Poisson(matrix[i][j]).
"""


"""2b. Construct transition probabilities between disease severities
There are three disease states: mild, severe and critical.
- Mild represents sub-hospitalization.
- Severe is hospitalization.
- Critical is ICU.

The key results of this section are:
- p_mild_severe: n_ages x 2 x 2 matrix. For each age and comorbidity state
    (length two bool vector indicating whether the individual has diabetes and/or
    hypertension), what is the probability of the individual transitioning from
    the mild to severe state.
- p_severe_critical, p_critical_death are the same for the other state transitions.

All of these probabilities are proportional to the base progression rate
for an (age, diabetes, hypertension) state which is stored in p_death_target
and estimated via logistic regression.
"""

"""
Overall probability of progressing to severe infection (hospitalization) for each age
Using the estimates from Verity et al 
https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext
"""
p_mild_severe_verity = np.zeros(n_ages)
p_mild_severe_verity[0:10] = 0
p_mild_severe_verity[10:20] = 0.0408
p_mild_severe_verity[20:30] = 1.04
p_mild_severe_verity[30:40] = 3.43
p_mild_severe_verity[40:50] = 4.25
p_mild_severe_verity[50:60] = 8.16
p_mild_severe_verity[60:70] = 11.8
p_mild_severe_verity[70:80] = 16.6
p_mild_severe_verity[80:] = 18.4
p_mild_severe_verity = p_mild_severe_verity/100


"""
Overall probability of ICU admission given severe infection for each age.
Using p(ICU)/p(ICU or hospitalized) from CDC, taking midpoint of intervals
https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm?s_cid=mm6912e2_w#T1_down
"""
icu_cdc = np.zeros(n_ages)
hosp_cdc = np.zeros(n_ages)
icu_cdc[:20] = 0
icu_cdc[20:45] = (2 + 4.2)/2
icu_cdc[45:55] = (5.4 + 10.4)/2
icu_cdc[55:65] = (4.7 + 11.2)/2
icu_cdc[65:75] = (8.1 + 18.8)/2
icu_cdc[75:] = (10.5+31.0	)/2
icu_cdc = icu_cdc/100

hosp_cdc[:20] = 0
hosp_cdc[20:45] = (14.3 + 20.8)/2
hosp_cdc[45:55] = (21.2 + 28.3)/2
hosp_cdc[55:65] = (20.5 + 30.1)/2
hosp_cdc[65:75] = (28.6 + 43.5)/2
hosp_cdc[75:] = (30.5 + 58.7)/2
hosp_cdc = hosp_cdc/100
overall_p_severe_critical = icu_cdc/(icu_cdc + hosp_cdc)
overall_p_severe_critical[:20] = 0

severe_critical_multiplier = overall_p_severe_critical / p_mild_severe_verity
critical_death_multiplier = overall_p_critical_death / p_mild_severe_verity
severe_critical_multiplier[:20] = 1
critical_death_multiplier[:20] = 1
# get the overall CFR for each age/comorbidity combination by running the logistic model
"""
Mortality model. We fit a logistic regression to estimate p_mild_death from
(age, diabetes, hypertension) to match the marginal mortality rates from TODO.
The results of the logistic regression are used to set the disease severity
transition probabilities.
"""
c_age = np.loadtxt('c_age_ifr.txt', delimiter=',').mean(axis=0)
"""float vector: Logistic regression weights for each age bracket."""
c_diabetes = np.loadtxt('c_diabetes_ifr.txt', delimiter=',').mean(axis=0)
"""float: Logistic regression weight for diabetes."""
c_hyper = np.loadtxt('c_hypertension_ifr.txt', delimiter=',').mean(axis=0)
"""float: Logistic regression weight for hypertension."""
intervals = np.loadtxt('comorbidity_age_intervals_ifr.txt', delimiter=',')

def age_to_interval(i):
    """Return the corresponding comorbidity age interval for a specific age.

    Args:
        i (int): age.

    Returns:
        int: index of interval containing i in intervals.
    """
    for idx, a in enumerate(intervals):
        if i >= a[0] and i < a[1]:
            return idx
    return idx

p_death_target = np.zeros((n_ages, 2, 2))
for i in range(n_ages):
    for diabetes_state in [0,1]:
        for hyper_state in [0,1]:
            if i < intervals[0][0]:
                p_death_target[i, diabetes_state, hyper_state] = 0
            else:
                p_death_target[i, diabetes_state, hyper_state] = scipy.special.expit(
                    c_age[age_to_interval(i)] + diabetes_state * c_diabetes +
                    hyper_state * c_hyper)






# n x 1 datatypes
r0_total = np.zeros((N_SIMS_PER_JOB,1))
ifr_total = np.zeros((N_SIMS_PER_JOB,1))
frac_death_older = np.zeros((N_SIMS_PER_JOB,1))

mse_list = np.zeros((N_SIMS_PER_JOB,1))

# n x T datatypes
#T = int(params['T'])
S_per_time = np.zeros((N_SIMS_PER_JOB, T))
E_per_time = np.zeros((N_SIMS_PER_JOB, T))
D_per_time = np.zeros((N_SIMS_PER_JOB, T))

Mild_per_time = np.zeros((N_SIMS_PER_JOB, T))
Severe_per_time = np.zeros((N_SIMS_PER_JOB, T))
Critical_per_time = np.zeros((N_SIMS_PER_JOB, T))
R_per_time = np.zeros((N_SIMS_PER_JOB, T))
Q_per_time = np.zeros((N_SIMS_PER_JOB, T))

r_0_over_time = np.zeros((N_SIMS_PER_JOB, T))
cfr_over_time = np.zeros((N_SIMS_PER_JOB, T))
fraction_over_70_time = np.zeros((N_SIMS_PER_JOB, T))
fraction_below_30_time = np.zeros((N_SIMS_PER_JOB, T))
median_age_time = np.zeros((N_SIMS_PER_JOB, T))

print('start loading pop')
if country == 'China' or country == 'Italy':
    pop_path = os.path.join(args.popdir, '{}_population_{}.pickle'.format(country, int(params['n'])))
    age, households, diabetes, hypertension, age_groups =  pickle.load(open(pop_path, 'rb'))
    
    
    age = age.astype(np.uint8)
    households = households.astype(np.int32)
    
    
    sector = np.zeros(age.shape[0], dtype=np.uint8)
    edu_sector = 0
    customer_facing = np.zeros(age.shape[0], dtype=np.uint8)
    age_groups_sector = numba.typed.List()
    age_groups_sector.append(np.array([],dtype=np.int32))
    sector_groups = None
elif country == 'NYC':
    age, households, diabetes, hypertension, age_groups, sector, \
    edu_sector, customer_facing, age_groups_sector \
    = pickle.load(open('{}_population_{}.pickle'.format(country, int(params['n'])), 'rb'))
    sector_groups = None
    age_groups_sector = numba.typed.List()
    age_groups_sector.append(np.array([],dtype=np.int32))
    age = age.astype(np.uint8)
    households = households.astype(np.int32)
    
    
    sector = np.zeros(age.shape[0], dtype=np.uint8)
    edu_sector = 0
    customer_facing = np.zeros(age.shape[0], dtype=np.uint8)

else:
    raise Exception('unknown country')
n_sectors = 1
# n x n_age_groups x T datatypes
infected_by_age_by_time = np.zeros((N_SIMS_PER_JOB, T, n_ages, n_sectors))
dead_by_age_by_time = np.zeros((N_SIMS_PER_JOB, T, n_ages, n_sectors))
recovered_by_age_by_time = np.zeros((N_SIMS_PER_JOB, T, n_ages, n_sectors))


pigc_vals_store = np.zeros(N_SIMS_PER_JOB)
mm_vals_store = np.zeros(N_SIMS_PER_JOB)
sd_vals_store = [0] * N_SIMS_PER_JOB
pinf_mult_vals_store = np.zeros(N_SIMS_PER_JOB)
infected_start_store = np.zeros(N_SIMS_PER_JOB)

# What dates are we simulating
for simnum in range(N_SIMS_PER_JOB):
    
    if be_bayesian:
        pigc_val = p_infect_given_contact_list[INDEX*N_SIMS_PER_JOB + simnum]
        mm_val = mortality_multiplier_list[INDEX*N_SIMS_PER_JOB + simnum]
        sd_val = start_date_list[INDEX*N_SIMS_PER_JOB + simnum]
        pinf_mult_val = pinf_mult_list[INDEX*N_SIMS_PER_JOB + simnum]
        n_infected_start_val = start_infected_list[INDEX*N_SIMS_PER_JOB + simnum]
        pigc_vals_store[simnum] = pigc_val
        mm_vals_store[simnum] = mm_val
        sd_vals_store[simnum] = sd_val
        pinf_mult_vals_store[simnum] = pinf_mult_val
        infected_start_store[simnum] = n_infected_start_val
        d0  = datetime.datetime.strptime(sd_val, '%m-%d-%Y').date()
        t_offset = (d0 - d_earliest_start).days
        
        print(pigc_val, mm_val, sd_val, pinf_mult_val)
    else:
        
        pigc_val = this_combo[0]
        mm_val = this_combo[1]
        sd_val = this_combo[2]
        pinf_mult_val = float(this_combo[3])
        
        print('combo',param_combo_index)
        print(this_combo)
        t_offset = 0
    
    d0  = datetime.datetime.strptime(sd_val, '%m-%d-%Y').date()
    # d_lockdown = date(2020, 4, 2) # Florida
    # d_lockdown = date(2020, 4, 1) # Mississippi
    
    T = float((d_end - d0).days + 1)
    params['T'] = T

    #these parameters are unused
    npi_sequence = np.zeros(int(params['T']), dtype=np.int32)
    npi_internal = []
    params['t_NPI_start'] = float((date(3020, 8, 8) - d0).days)   
    print('NPIs start on ' + str(params['t_NPI_start']))  
    
    #day that travel ban from china was effective -- no importations were used in these experiments
    d_end_imports = date(2020, 2, 2)
    
    # Parse real data for computing mse during sweep
    data = pd.read_csv('validation_data/italy/lombardy_data_deaths.csv')
    dates = []
    actual_deaths = []
    for i in range(len(data)):
        dates.append(datetime.datetime.strptime(data['Date'][i], '%m/%d/%y %H:%M').date())
        actual_deaths.append(data['Deaths'][i])
    
    time_from_d0 = []
    for i in range(len(dates)):
        time_from_d0.append((dates[i] - d0).days)
    
    
    """1. SET FIXED SIMULATION PARAMETERS
    """
    
    
    
    
    
    
    """int: Number of timesteps"""
    
    params['initial_infected_fraction'] = n_infected_start_val/params['n']
    params['t_lockdown'] = float((d_lockdown - d0).days)
    params['t_lockdown_release'] = float((d_lockdown_release - d0).days)
    params['t_school_lockdown'] = float((d_school_lockdown - d0).days)
    params['close_schools_lockdown'] = float(True)
    if 'close_schools_lockdown' in input_dict and input_dict['close_schools_lockdown'] == 'False':
        params['close_schools_lockdown'] = float(False)
    
    try:
        factor = input_dict['lockdown_factor']
    except KeyError:
        factor = args.lockdown
        print('lockdown factor not specified in json, setting to ' + str(factor))
        
    params['lockdown_factor'] = factor
    if 'second_lockdown_factor' in input_dict:
        params['second_lockdown_factor'] = input_dict['second_lockdown_factor']
        d_second_lockdown = datetime.datetime.strptime(input_dict['d_second_lockdown'], '%m-%d-%Y').date()
        params['t_second_lockdown'] = float((d_second_lockdown - d0).days)
    lockdown_factor_age = ((0, 14, factor), (15, 24, factor), (25, 39, factor), (40, 69, factor), (70, 100, factor))
    lockdown_factor_age = np.array(lockdown_factor_age)
        
    
    d_stay_home_start = date(2030, 8, 8) #no stay home in the US
    if 'd_stay_home' in input_dict:
        d_stay_home_start = datetime.datetime.strptime(input_dict['d_stay_home'], '%m-%d-%Y').date()
    params['t_stayhome_start'] = float((d_stay_home_start - d0).days)
    print('stay home', params['t_stayhome_start'])
    
    #https://www.medrxiv.org/content/10.1101/2020.02.17.20024075v1.full.pdf
    params['mean_total_imports'] = float(0) 
    params['t_end_imports'] = float((d_end_imports - d0).days)
    

    
    params['p_infect_given_contact'] = pigc_val
    """float: Probability of infection given contact between two individuals
    This is currently set arbitrarily and will be calibrated to match the empirical r0
    """
    
    params['mortality_multiplier'] = mm_val
    dmult_mult = input_dict['dmult_mult']
    
    #calibrate the probability of the severe -> critical transition to match the
    #overall CFR for each age/comorbidity combination
    #age group, diabetes (0/1), hypertension (0/1)
    progression_rate = np.zeros((n_ages, 2, 2))
    p_mild_severe = np.zeros((n_ages, 2, 2))
    """float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from mild to severe state."""
    p_severe_critical = np.zeros((n_ages, 2, 2))
    """float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from severe to critical state."""
    p_critical_death = np.zeros((n_ages, 2, 2))
    """float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from critical to dead state."""
    
    for i in range(n_ages):
        for diabetes_state in [0,1]:
            for hyper_state in [0,1]:
                progression_rate[i, diabetes_state, hyper_state] = (p_death_target[i, diabetes_state, hyper_state]
                                                                    / (severe_critical_multiplier[i]
                                                                       * critical_death_multiplier[i])) ** (1./3)
                p_mild_severe[i, diabetes_state, hyper_state] = progression_rate[i, diabetes_state, hyper_state]
                p_severe_critical[i, diabetes_state, hyper_state] = severe_critical_multiplier[i]*progression_rate[i, diabetes_state, hyper_state]
                p_critical_death[i, diabetes_state, hyper_state] = critical_death_multiplier[i]*progression_rate[i, diabetes_state, hyper_state]
    #no critical cases under 20 (CDC)
    p_critical_death[:20] = 0
    p_severe_critical[:20] = 0
    #for now, just cap 80+yos with diabetes and hypertension
    p_critical_death[p_critical_death > 1] = 1
    p_mild_severe *= params['mortality_multiplier']**(1/3)
    p_severe_critical *= params['mortality_multiplier']**(1/3)
    p_critical_death *= params['mortality_multiplier']**(1/3)
    p_mild_severe[60:] *= dmult_mult**(1/3)
    p_severe_critical[60:] *= dmult_mult**(1/3)
    p_critical_death[60:] *= dmult_mult**(1/3)

    p_mild_severe[p_mild_severe > 1] = 1
    p_severe_critical[p_severe_critical > 1] = 1
    p_critical_death[p_critical_death > 1] = 1

    """
    float: increase probability of death for all ages and comorbidities by this amount
    """
    
    params['contact_tracing'] = float(False)
    params['p_trace_outside'] = 1.0
    params['p_trace_household'] = 0.75
    d_tracing_start = date(2020, 2, 10)
    params['t_tracing_start'] = float((d_tracing_start - d0).days)
    """
    Whether contact tracing happens, and if so the probability of successfully 
    identifying each within and between household infected individual
    """        
    
    

    pinf_mult = np.ones(n_ages)
    pinf_mult[60:] = pinf_mult_val
    
    
    params['seed']+=1

    S, E, Mild, Documented, Severe, Critical, R, D, Q, num_infected_by,time_documented, \
    time_to_activation, time_to_death, time_to_recovery, time_critical, time_exposed, num_infected_asympt,\
        age, time_infected, time_to_severe, R_last, D_last, infected_by_age_by_time_i, dead_by_age_by_time_i, recovered_by_age_by_time_i,\
        mild_sector_time_i, severe_sector_time_i, critical_sector_time_i, isolated_sector_time_i, recovered_sector_time_i, dead_sector_time_i=\
        run_model(int(params['seed']), households, age, age_groups, diabetes, hypertension, \
                     other_contact, work_contact, customer_contact, school_contact, sector, edu_sector, customer_facing, age_groups_sector,\
                     p_mild_severe, p_severe_critical, p_critical_death, mean_time_to_isolate_factor, lockdown_factor_age, fraction_stay_home, params, \
                     npi_internal, npi_sequence, sector_groups, pinf_mult)

   
    S_per_time[simnum, t_offset:] = S[:]
    E_per_time[simnum, t_offset:] = E[:]
    D_per_time[simnum, t_offset:] = D[:]

    Mild_per_time[simnum, t_offset:] = Mild[:]
    Severe_per_time[simnum, t_offset:] = Severe[:]
    Critical_per_time[simnum, t_offset:] = Critical[:]
    R_per_time[simnum, t_offset:] = R[:]
    Q_per_time[simnum, t_offset:] = Q[:]




    if country == 'Italy':
        time_cutoff = 20
    elif country == 'China':
        time_cutoff = 55
    elif country == 'NYC':
        time_cutoff = (d_lockdown - d0).days - 20
        if time_cutoff < 5:
            time_cutoff = 5
    r0_total[simnum] = num_infected_by[np.logical_and(time_exposed <= time_cutoff, time_exposed > 0)].mean()
    exposed_early = np.logical_and(time_exposed <= time_cutoff, time_exposed > 0)
    print('sample size', exposed_early.sum())
    exposed_not_late = np.logical_and(time_exposed <= T - 40, time_exposed > 0)
    ifr_total[simnum] = D_last[exposed_not_late].sum()/(D_last[exposed_not_late].sum() + R_last[exposed_not_late].sum())
    deaths_over_65 = D_last[age >= 60].sum()
    deaths_under_65 = D_last[age < 60].sum()
    frac_death_older[simnum] = deaths_over_65/(deaths_over_65 + deaths_under_65)
    print('r0_total after')
    print(r0_total[simnum])
    print('ifr total after')
    print(ifr_total[simnum])
    print('frac_death_older')
    print(frac_death_older[simnum])
    
    print('AR over 60', (time_exposed > 0)[age >= 60].mean())
    print('AR under 60', (time_exposed > 0)[age < 60].mean())

    print('S')
    print(S_per_time[simnum][-1])
    print('Exposed')
    print(E_per_time[simnum][-1])
    print('R')
    print(R_per_time[simnum][-1])
    print('Dead')
    print(D_per_time[simnum][-1])

    print('Mild_per_time')
    print(Mild_per_time[simnum][-1])
    print('Severe_per_time')
    print(Severe_per_time[simnum][-1])
    print('Critical_per_time')
    print(Critical_per_time[simnum][-1])
    print('Q')
    print(Q_per_time[simnum][-1])

    infected_by_age_by_time[simnum, t_offset:] = infected_by_age_by_time_i[:]
    dead_by_age_by_time[simnum, t_offset:] = dead_by_age_by_time_i[:]
    recovered_by_age_by_time[simnum, t_offset:] = recovered_by_age_by_time_i[:]





print('r0_total after')
print(r0_total.mean())
print('ifr total after')
print(ifr_total.mean())
print('frac_death_older')
print(frac_death_older.mean())



path = os.path.join(EXP_ROOT_DIR, sim_name)

if be_bayesian:
    fname = '%s_bayesian_n%s_i%s'%(sim_name, params['n'], INDEX)
    fname += '_%s.hdf'
else:
    fname = '%s_paramsweep_n%s_i%s_N%s_p%s_m%s_s%s_l%s_pim%s'%(sim_name, params['n'], param_combo_index, TRIAL_NUMBER, pigc_val, mm_val, sd_val, factor, pinf_mult_val)
    fname += '_%s.hdf'

fname = os.path.join(path, fname)

datatypes_n_1 = [
    {
        'name':'r0_tot', 
        'data':r0_total
    },
    {
        'name':'mse',
        'data':mse_list
    },
    {
        'name':'ifr_tot',
        'data':ifr_total
    },
    {
        'name':'frac_death_older',
        'data':frac_death_older
    }


]

if be_bayesian:
    datatypes_n_1 += [
        {
            'name':'pigc_vals_store', 
            'data':pigc_vals_store
        },
        {
            'name':'mm_vals_store',
            'data':mm_vals_store
        },
        {
            'name':'sd_vals_store',
            'data':sd_vals_store
        },
        {
            'name':'pinf_mult_vals_store',
            'data':pinf_mult_vals_store
        },        
        {
            'name':'infected_start_store',
            'data':infected_start_store
        }

    ]

datatypes_n_t = [
    {
        'name':'susceptible', 
        'data':S_per_time
    },
    {
        'name':'exposed', 
        'data':E_per_time
    },
    {
        'name':'deaths', 
        'data':D_per_time
    },
    {
        'name':'mild', 
        'data':Mild_per_time
    },
    {
        'name':'severe', 
        'data':Severe_per_time
    },
    {
        'name':'critical', 
        'data':Critical_per_time
    },
    {
        'name':'recovered', 
        'data':R_per_time
    },
    {
        'name':'quarantine', 
        'data':Q_per_time
    },

#    {
#        'name':'r0_time', 
#        'data':r_0_over_time
#    },
#    {
#        'name':'cfr_time', 
#        'data':cfr_over_time
#    },
#    {
#        'name':'frac_over_70_time', 
#        'data':fraction_over_70_time
#    },
#    {
#        'name':'frac_below_30_time', 
#        'data':fraction_below_30_time
#    },
#    {
#        'name':'median_age_time', 
#        'data':median_age_time
#    }
    
]

# These will be the biggest storage burden. Cut where you can.
datatypes_n_a_t = [
#    {
#        'name':'infected_age_time', 
#        'data':infected_by_age_by_time
#    },
#    # {
#    #     'name':'cfr_age_time',
#    #     'data':CFR_by_age_by_time
#    # },
#    {
#        'name':'dead_age_time',
#        'data':dead_by_age_by_time
#    },
#    {
#        'name':'recovered_age_time',
#        'data':recovered_by_age_by_time
#    }

]

datatypes_n_a = [

]

print(fname)
for d in datatypes_n_1:
#    print(d)
    df = pd.DataFrame(d['data'])
    df.to_hdf(fname%d['name'], key='values')


for d in datatypes_n_t:
#    print(d)
    df = pd.DataFrame(d['data'])
    df.to_hdf(fname%d['name'], key='values')


for d in datatypes_n_a_t:
    # Flatten the data -- remember to unflatten for analysis
    d['data'] = d['data'].reshape(N_SIMS_PER_JOB, -1)
    df = pd.DataFrame(d['data'])
    df.to_hdf(fname%d['name'], key='values')

for d in datatypes_n_a:
    df = pd.DataFrame(d['data'])
    df.to_hdf(fname%d['name'], key='values')


FILEEND = time.time()

print('Full run took %ss'%(FILEEND - FILESTART))



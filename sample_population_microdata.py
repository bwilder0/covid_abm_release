import pandas as pd
import numpy as np
from numba import jit
from numba.typed import List
import pickle
from sample_comorbidities import sample_joint_comorbidities
import argparse
import os
import json

parser = argparse.ArgumentParser(description='Sample IPUMS microdata for whole US or for individual state.')
parser.add_argument('--country', action='store_true', default=True, help='Run for whole US')
parser.add_argument('--load_population_microcodes', action='store_true', default=False, help='Run for whole US')
parser.add_argument('-s','--state', type=str, help='State to parse. If set, will override --country=True and only sample state.')
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling.')
parser.add_argument('npi_dir', type=str, help='Must specify an output dir for all intermediate files.')
args = parser.parse_args()

if not args.country and args.state is None:
    print('Must specify --country=True or choose a state with, e.g., --state Florida')
    exit()

state_dict = json.load(open('inputs/state_codes.json', 'r'))

implemented_state_list = list(state_dict.keys()) + ['Nyc', None]
#implemented_state_list = ['Mississippi', 'Florida', 'Nyc', 'Georgia', 'Massachusetts', 'Illinois', None]
if args.state is not None and args.state.title() not in implemented_state_list:
    print('{} has not been implemented yet.'.format(args.state.title()))
    exit()


def classify_occupation_consumer_facing(code):
    if code == 0:
        return False
    # Management
    elif code < 500:
        return False
    # Business and Financial, customer-facing
    elif (630 <= code < 640 or 725 <= code < 734 or
          810 <= code < 820 or 850 <= code < 860 or
          910 <= code < 960):
        return True
    # Psychologists (social science customer-facing)
    elif 1820 <= code < 1830:
        return True
    # Business and Financial, Computer and Engineering
    elif code < 2001:
        return False
    # Community and Social Service
    elif code < 2100:
        return True
    # Lawyers and judges (legal customer-facing)
    elif 2100 <= code < 2105 or 2110 <= code < 2145:
        return True
    # Legal
    elif code < 2205:
        return False
    # Education
    elif code < 2600:
        return True
    # Journalists and photographers (media customer-facing)
    elif 2810 <= code < 2825 or 2910 <= code < 2920:
        return True
    # Art, design, entertainment, sports, and media
    elif code < 3000:
        return False
    # "healthcare technicians", supervisors
    elif (3300 <= code < 3310 or
          3515 <= code < 3520 or 3646 <= code < 3647 or
          3700 <= code < 3740):
        return False
    # healthcare, healthcare support, protective service
    elif code < 4000:
        return True
    # "management" and "technical" aspects of food
    elif 4000 <= code < 4040 or 4140 <= code < 4150:
        return False
    # food preparation and serving
    elif code < 4200:
        return True
    # Building and grounds maintenance
    elif code < 4330:
        return False
    # animal care
    elif 4340 <= code < 4400:
        return False
    # Personal care and service
    elif code < 4700:
        return True
    # sales supervisors
    elif 4700 <= code < 4720:
        return False
    # sales
    elif code < 5000:
        return True
    # Office customer facing
    elif (5230 <= code < 5260 or 5300 <= code < 5350 or
          5400 <= code < 5420 or 5500 <= code < 5521 or
          5540 <= code < 5550 or 5700 <= code < 5800):
        return True
    # office, farming, construction, installation, production
    elif code < 9005:
        return False
    # transportation customer-facing
    elif (9050 <= code < 9150 or
          9350 <= code < 9410 or 9415 <= code < 9430):
        return True
    # transportation, material moving, military
    else:
        return False


def census_to_bea(x, bea_to_idx, naics_to_bea, census_naics):
    if x in naics_to_bea:
        return bea_to_idx[naics_to_bea[x]]
    else:
        naics = census_naics[census_naics['INDNAICS CODE'] == x]['2002 NAICS EQUIVALENT'].to_numpy()[0]
        if naics in naics_to_bea:
            bea = naics_to_bea[naics]
        elif naics[:4] in naics_to_bea:
            bea = naics_to_bea[naics[:4]]
        elif naics[:3] in naics_to_bea:
            bea = naics_to_bea[naics[:3]]
        elif naics[:2] in naics_to_bea:
            bea = naics_to_bea[naics[:2]]
        else:
            raise Exception('not found: ' + naics)
        return bea_to_idx[bea]
    

 

@jit(nopython=True)
def make_households(n, hh_size, hh_start, hh_sequence, households, max_household_size, age, micro_ages, sector, microdata_sector, microdata_customer, customer_facing):
    num_generated = 0
    num_hh_generated = 0
    while num_generated < n - max_household_size: #20 = max household size in the data
        choice = hh_sequence[num_hh_generated]
        if int(num_generated/1e6) != int((num_generated + hh_size[choice])/1e6):
            print(num_generated)
        for i in range(hh_size[choice]):
            age[num_generated + i] = micro_ages[hh_start[choice]+i]
            sector[num_generated + i] = microdata_sector[hh_start[choice]+i]
            customer_facing[num_generated + i] = microdata_customer[hh_start[choice]+i]
        generated_this_step = hh_size[choice]
        for i in range(num_generated, num_generated+generated_this_step):
            curr_pos = 0
            for j in range(num_generated, num_generated+generated_this_step):
                if i != j:
                    households[i, curr_pos] = j
                    curr_pos += 1
        num_generated += generated_this_step
        num_hh_generated += 1
    
    singletons = np.where(hh_size == 1)[0]
    for i in range(n - num_generated):
        choice = np.random.choice(singletons)
        age[num_generated + i] = micro_ages[hh_start[choice]]
        sector[num_generated + i] = microdata_sector[hh_start[choice]]
        customer_facing[num_generated + i] = microdata_customer[hh_start[choice]]
    return households, age



def main(state, seed, load_microdata_codes, DB_pop=False):

    np.random.seed(seed)

    print("Reading ipums data... ",end='')
    microdata = pd.read_csv('../covid_data/ipums_us.csv')
    print("Done.")


    print(state)
    if state is not None and state != 'NYC':
        n =  int(open('validation_data/US/{}/{}_population.csv'.format(state,state),'r').read())
    elif state == 'NYC':
        n = int(8.4e6)
    else:
        n = int(330e6)

    # https://international.ipums.org/international/resources/misc_docs/geolevel1.pdf
    # TODO: make a file with all this info and parse
    if state != 'NYC':
        statenum = state_dict[state]
    # if state == 'Florida':
    #     statenum = 840012
    # elif state == 'Mississippi':
    #     statenum = 840028
    # elif state == 'Georgia':
    #     statenum = 840013
    # elif state == 'Massachusetts':
    #     statenum = 840025
    # elif state == 'Illinois':
    #     statenum = 840017

    if state is not None and state != 'NYC':
        microdata = microdata[microdata.GEOLEV1 == statenum]
        microdata = microdata.reset_index()
    elif state == 'NYC':
        puma_descriptions = pd.read_csv('puma_descriptions.csv')
        nyc = puma_descriptions[puma_descriptions['PUMA NAME'].str.contains('NYC')].PUMA5CE
        nyc = [int(str(360) + str(x)) for x in nyc]
        microdata = microdata[microdata.GEO2_US2015.isin(nyc)]
        microdata = microdata.reset_index()
        
    hh_id = microdata.SERIAL.to_numpy()
    n_hh = np.unique(hh_id).shape[0]
    hh_start = np.zeros(n_hh)
    hh_weight = np.zeros(n_hh)
    curr_hh = 0
    curr_hh_id = hh_id[0]
    hh_weight[0] = microdata.HHWT[0]
    for i in range(hh_id.shape[0]):
        if hh_id[i] != curr_hh_id:
            curr_hh += 1
            hh_start[curr_hh] = i
            curr_hh_id = hh_id[i]
            hh_weight[curr_hh] = microdata.HHWT[i]

    hh_weight = hh_weight/hh_weight.sum()

    hh_size = np.zeros(n_hh)
    hh_size[:-1] = hh_start[1:] - hh_start[:-1]
    hh_size[-1] = microdata.shape[0] - hh_start[-1]
    hh_size = hh_size.astype(np.int32)
    max_household_size = int(hh_size.max())
    hh_start = hh_start.astype(np.int32)
    micro_ages = microdata.AGE.to_numpy().astype(np.int32)

#    a = pd.read_csv('naics_bea.csv')
#    a = a[1::2].reset_index()
#
#    bea_to_idx = {a.industry[i]:i for i in a.index}
    bea_to_idx = json.load(open('bea_to_idx.json', 'r'))
    naics_to_bea = json.load(open('naics_to_bea.json', 'r'))
#    naics_to_bea = {}
#    for i in a.index:
#        for s in a.naics_code[i].split(','):
#            naics_to_bea[s.strip()] = a.industry[i]

    census_naics = pd.read_csv('census_to_naics.csv')





    microdata_sector = np.zeros(microdata.shape[0], dtype=np.uint8)
    microdata_customer = np.zeros(microdata.shape[0], dtype=np.bool8)  



    age = np.zeros(n, dtype=np.uint8)
    sector = np.zeros(n, dtype=np.uint8)
    customer_facing = np.zeros(n, dtype=np.bool8)

    hh_sequence = np.random.choice(n_hh, size=n, p=hh_weight)

    households = np.zeros((n, max_household_size), dtype=np.int32)
    households[:] = -1
    print('allocated')
    if not load_microdata_codes:
        total_num = microdata.index.shape
        for i in microdata.index:
            print (i,total_num)

            if DB_pop:
                occ_code = microdata.US2015A_OCC[i]
                microdata_customer[i] = classify_occupation_consumer_facing(occ_code)
                ind_code = microdata.US2015A_INDNAICS[i]
                microdata_sector[i] = census_to_bea(ind_code, bea_to_idx, naics_to_bea, census_naics)
            else:
                if microdata['EMPSTAT'][i] == 1:
                    occ_code = microdata.US2015A_OCC[i]
                    microdata_customer[i] = classify_occupation_consumer_facing(occ_code)
                    ind_code = microdata.US2015A_INDNAICS[i]
                    microdata_sector[i] = census_to_bea(ind_code, bea_to_idx, naics_to_bea, census_naics)
                else:
                    microdata_customer[i] = False
                    microdata_sector[i] = 0
        
        #split 52M2 into 523 and 525 and 531 into HS and ORE
        curr_523 = np.where(microdata_sector == bea_to_idx['523'])[0]
        to_reassign = np.random.choice(curr_523, int(0.5*curr_523.shape[0]), replace=False)
        microdata_sector[to_reassign] = bea_to_idx['525']
   
        curr_HS = np.where(microdata_sector == bea_to_idx['HS'])[0]
        to_reassign = np.random.choice(curr_HS, int(0.5*curr_HS.shape[0]), replace=False)
        microdata_sector[to_reassign] = bea_to_idx['ORE']
        
        data = (microdata_customer, microdata_sector)
        
        with open('ipums_us_microdata_codes.pickle', 'wb') as f:
            pickle.dump(data, f)
    else:
        print('loading microdata')
        microdata_customer, microdata_sector = pickle.load(open('ipums_us_microdata_codes.pickle', 'rb'))

    print('done setting/loading microdata codes')

       


    print("Making households... ")
    households, age = make_households(n, hh_size, hh_start, hh_sequence, households, max_household_size, age, micro_ages, sector, microdata_sector, microdata_customer, customer_facing)
    print("Done.")
    n_age = 101

    # TODO: currently the largest bottleneck, 
    #       consider hardcoding, if we always expect all ages and sectors.
    print("Creating age group sector array... ")
    n_sector = np.unique(microdata_sector).shape[0]
    age_groups = tuple([np.where(age == i)[0] for i in range(0, n_age)])
    age_groups_sector = []
    sector_groups = []
    for j in range(n_sector):
        print(j)
        sector_group_j = np.where(sector == j)[0]
#        age_groups_sector.extend([np.where(np.logical_and(age == i, sector == j))[0] for i in range(0, n_age)])
        age_groups_sector.extend([sector_group_j[age[sector_group_j] == i] for i in range(0, n_age)])
        sector_groups.append(sector_group_j)
    print("Done.")

    # TODO: Sample from state specific comorbidity distributions
    print("Sampling comorbidities... ")
    diabetes, hypertension = None, None
    if state is not None:
        diabetes, hypertension = sample_joint_comorbidities(age, state)
    else:
        diabetes, hypertension = sample_joint_comorbidities(age, 'US')

    print("Done.")

    edu_sector = bea_to_idx['61A']

    print("Saving... ")
    if state is None:
        pickle.dump((age, households, diabetes, hypertension, age_groups, sector, edu_sector, customer_facing, age_groups_sector, sector_groups), open('/n/holylfs/LABS/tambe_lab/bwilder/US_population_{}.pickle'.format(int(n)), 'wb'), protocol=4)

    else:
        pickle.dump((age, households, diabetes, hypertension, age_groups, sector, edu_sector, customer_facing, age_groups_sector, sector_groups), open(os.path.join(args.npi_dir,'{}_population_{}.pickle'.format(state, int(n))), 'wb'), protocol=4)
    print("Done.")

load_population_microcodes = args.load_population_microcodes
state = args.state
if state != None and state != 'NYC':
    state = state.title()

DB_pop = True
main(state,args.seed,load_population_microcodes,DB_pop)


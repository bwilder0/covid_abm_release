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
import threading
import time
import queue


@jit(nopython=True)
def get_isolation_factor(age, mean_time_to_isolate_factor):
    for i in range(len(mean_time_to_isolate_factor)):
        if age >= mean_time_to_isolate_factor[i, 0] and age <= mean_time_to_isolate_factor[i, 1]:
            return mean_time_to_isolate_factor[i, 2]
    return 1

@jit(nopython=True)
def get_lockdown_factor_age(age, lockdown_factor_age):
    for i in range(len(lockdown_factor_age)):
        if age >= lockdown_factor_age[i, 0] and age <= lockdown_factor_age[i, 1]:
            return lockdown_factor_age[i, 2]
    return 1

@jit(nopython=True)
def do_contact_tracing(i, infected_by, p_trace_outside, Q, S, t, households, p_trace_household, Documented, time_documented, traced):
    #trace contacts within the household
    t_last = (t+1)%2
    for j in range(households.shape[1]):
        contact = households[i, j]
        if contact == -1:
            break
        if not S[t_last, contact] and not traced[contact] and np.random.rand() < p_trace_household:
            Q[t, contact] = True
            Documented[t, contact] = True
            traced[contact] = True
            time_documented[contact] = t
            do_contact_tracing(contact, infected_by, p_trace_outside, Q, S, t, households, p_trace_household, Documented, time_documented, traced)
    #trace outside of household contacts
    for j in range(infected_by.shape[1]):
        contact = infected_by[i, j]
        if contact == -1:
            break
        if np.random.rand() < p_trace_outside:
            Q[t, contact] = True
            Documented[t, contact] = True
            time_documented[contact] = t
            traced[contact] = True
            do_contact_tracing(contact, infected_by, p_trace_outside, Q, S, t, households, p_trace_household, Documented, time_documented, traced)


def accumulate_into_queue(q, S, E, Mild, Severe, Critical, R, D, Q, Work_status, age, n_age, sector, n_sector, start_block, end_block):
    q.put(accumulate(S, E, Mild, Severe, Critical, R, D, Q, Work_status, age, n_age, sector, n_sector, start_block, end_block))

@jit(nopython=True,nogil=True)
def accumulate(S, E, Mild, Severe, Critical, R, D, Q, Work_status, age, n_age, sector, n_sector, start_block, end_block): 
    s_total = 0
    e_total = 0
    mild_total = 0
    severe_total = 0
    critical_total = 0
    r_total = 0
    d_total = 0
    q_total = 0
    symptomatic_age = np.zeros((n_age, n_sector), dtype=np.int32)
    recovered_age = np.zeros((n_age, n_sector), dtype=np.int32)
    dead_age = np.zeros((n_age, n_sector), dtype=np.int32)
    
    mild_sector = np.zeros((n_sector, 3), dtype=np.int32)
    severe_sector = np.zeros((n_sector, 3), dtype=np.int32)
    critical_sector = np.zeros((n_sector, 3), dtype=np.int32)
    isolated_sector = np.zeros((n_sector, 3), dtype=np.int32)
    recovered_sector = np.zeros((n_sector, 3), dtype=np.int32)
    dead_sector = np.zeros((n_sector, 3), dtype=np.int32)
    
    for i in range(start_block, end_block):
        s_total += S[i]
        e_total += E[i]
        mild_total += Mild[i]
        severe_total += Severe[i]
        critical_total += Critical[i]
        r_total += R[i]
        d_total += D[i]
        q_total += Q[i]
        symptomatic_age[age[i], sector[i]] += Mild[i] + Severe[i] + Critical[i]
        recovered_age[age[i], sector[i]] += R[i]
        dead_age[age[i], sector[i]] += D[i]
        
        mild_sector[sector[i], Work_status[i]] += Mild[i]
        severe_sector[sector[i], Work_status[i]] += Severe[i]
        critical_sector[sector[i], Work_status[i]] += Critical[i]
        isolated_sector[sector[i], Work_status[i]] += Q[i]
        recovered_sector[sector[i], Work_status[i]] += R[i]
        dead_sector[sector[i], Work_status[i]] += D[i]
        
    return s_total, e_total, mild_total, severe_total, critical_total, r_total, d_total, q_total, symptomatic_age, recovered_age, dead_age, \
            mild_sector, severe_sector, critical_sector, isolated_sector, recovered_sector, dead_sector
    
@jit(nopython=True)
def assign_stay_home(seed, fraction_stay_home, Home_real, age):
    np.random.seed(seed)
    for i in range(age.shape[0]):
        if np.random.rand() < fraction_stay_home[age[i]]:
            Home_real[i] = True


@jit(nopython=True)
def reassign_work_status(to_reassign, Work_status, deltas, idxs):
    proportion_first = deltas[0]/(deltas[0] + deltas[1])
    num_first = int(proportion_first*to_reassign.shape[0])
    for i in range(num_first):
        Work_status[to_reassign[i]] = idxs[0]
    for i in range(num_first, to_reassign.shape[0]):
        Work_status[to_reassign[i]] = idxs[1]
        

def run_model(seed, households, age, age_groups, diabetes, hypertension,\
 other_contact, work_contact, customer_contact, school_contact, sector, edu_sector, customer_facing, age_groups_sector,\
 p_mild_severe, p_severe_critical, p_critical_death, mean_time_to_isolate_factor, \
 lockdown_factor_age, fraction_stay_home, params, npis, npi_sequence, sector_groups, pinf_mult):
    print('run_model')
    """Run the SEIR model to completion.

    Args:
        seed (int): Random seed.
        households: Household structure.
        age (int vector of length n): Age of each individual.
        diabetes (bool vector of length n): Diabetes state of each individual.
        hypertension (bool vector of length n): Hypertension state of each
            individual.

    Returns:
        S (bool T x n matrix): Matrix where S[i][j] represents
            whether individual i was in the Susceptible state at time j.
        E (bool T x n matrix): same for Exposed state.
        Mild (bool T x n matrix): same for Mild state.
        Severe (bool T x n matrix): same for Severe state.
        Critical (bool T x n matrix): same for Critical state.
        R (bool T x n matrix): same for Recovered state.
        D (bool T x n matrix): same for Dead state.
        Q (bool T x n matrix): same for Quarantined state.
        num_infected_by (n vector): num_infected_by[i] is the number of individuals
            infected by individual i.
        time_to_activation (n vector): TODO
        time_to_death (n vector):  TODO
        time_to_recovery: TODO
        time_critical: TODO
        time_exposed: TODO
    """
    start_all = time.time()
    n = int(params['n'])
    n_threads = int(params['n_threads'])
    n_ages = int(params['n_ages'])
    n_sector = np.unique(sector).shape[0]
    T = int(params['T'])
    t_lockdown = int(params['t_lockdown'])
    if 't_second_lockdown' in params:
        t_second_lockdown = int(params['t_second_lockdown']) 
    else:
        t_second_lockdown = 10000
    print(t_second_lockdown)
    t_lockdown_release = int(params['t_lockdown_release'])
    t_school_lockdown = int(params['t_school_lockdown'])
    lockdown_factor = params['lockdown_factor']
    if 'second_lockdown_factor' in params:
        second_lockdown_factor = params['second_lockdown_factor']
    else:
        second_lockdown_factor = 1
    t_stayinghome_start = int(params['t_stayhome_start'])
    contact_tracing = bool(params['contact_tracing'])
    t_tracing_start = int(params['t_tracing_start'])
    t_stayinghome_start = int(params['t_stayhome_start'])
    time_to_activation_mean = params['time_to_activation_mean']
    time_to_activation_std = params['time_to_activation_std']
    t_end_imports = int(params['t_end_imports'])
    mean_total_imports = params['mean_total_imports']
    tracing_enabled = False
    if contact_tracing:
        tracing_enabled = True
        contact_tracing = False
    initial_infected_fraction = params['initial_infected_fraction']
    np.random.seed(seed)
    S = np.zeros((2, n), dtype=np.bool8)
    E = np.zeros((2, n), dtype=np.bool8)
    Mild = np.zeros((2, n), dtype=np.bool8)
    Documented = np.zeros((2, n), dtype=np.bool8)
    Severe = np.zeros((2, n), dtype=np.bool8)
    Critical = np.zeros((2, n), dtype=np.bool8)
    R = np.zeros((2, n), dtype=np.bool8)
    D = np.zeros((2, n), dtype=np.bool8)
    Q = np.zeros((2, n), dtype=np.bool8)
    traced = np.zeros((n), dtype=np.bool8)
    Home_real = np.zeros(n, dtype=np.bool8)
    Home_real[:] = False
    Work = np.zeros(n, dtype=np.bool8)
    Work[:] = True
    #0: working normally 1: WFH 2: laid off
    Work_status = np.zeros(n, dtype=np.uint8)
    
    school_contact = np.copy(school_contact)
    other_contact = np.copy(other_contact)
    work_contact = np.copy(work_contact)
    customer_contact = np.copy(customer_contact)

    school_contact_init = np.copy(school_contact)
    other_contact_init = np.copy(other_contact)
    work_contact_init = np.copy(work_contact)
    customer_contact_init = np.copy(customer_contact)
    
    assign_stay_home(seed, fraction_stay_home, Home_real, age)
            
    dummy_Home = np.zeros(n, dtype=np.bool8)
    dummy_Home[:] = False
    Home = dummy_Home
    initial_infected = functions.resevoir_sample(n, int(initial_infected_fraction*n))
    S[0] = True
    E[0] = False
    R[0] = False
    D[0] = False
    Mild[0] = False
    Documented[0]=False
    Severe[0] = False
    Critical[0] = False

    if tracing_enabled:
        infected_by = np.zeros((n, 40), dtype=np.int32)
        infected_by[:] = -1
    else:
        infected_by = np.zeros((2, 2), dtype=np.int32)
    
    time_exposed = np.zeros(n, dtype=np.int16)
    time_infected = np.zeros(n, dtype=np.int16)
    time_severe = np.zeros(n, dtype=np.int16)
    time_critical = np.zeros(n, dtype=np.int16)
    time_documented=np.zeros(n, dtype=np.int16)
    time_exposed[:] = -1
    #total number of infections caused by every individual, -1 if never become infectious
    #assumes no one causes more than 127 infections
    num_infected_by = np.zeros(n, dtype=np.int8)
    num_infected_by_outside = np.zeros(n, dtype=np.int8)
    num_infected_asympt = np.zeros(n, dtype=np.int8)
    num_infected_by[:] = -1
    num_infected_by_outside[:] = -1
    num_infected_asympt[:] = -1
    #assumes that simulation runs at most 256 days, otherwise need uint16
    time_to_severe = np.zeros(n, dtype=np.uint16)
    time_to_recovery = np.zeros(n, dtype=np.uint16)
    time_to_critical = np.zeros(n, dtype=np.uint16)
    time_to_death = np.zeros(n, dtype=np.uint16)
    time_to_isolate = np.zeros(n, dtype=np.uint16)
    time_to_activation = np.zeros(n, dtype=np.uint16)

#    time_to_documented= np.zeros(n)
#    previously_picked_this_round = np.zeros(n, dtype=np.bool8)


    # For saving condensed results
    S_per_time = np.zeros(T)
    E_per_time = np.zeros(T)
    D_per_time = np.zeros(T)

    Mild_per_time = np.zeros(T)
    Severe_per_time = np.zeros(T)
    Critical_per_time = np.zeros(T)
    R_per_time = np.zeros(T)
    Q_per_time = np.zeros(T)

    # n x n_age_groups x T datatypes
    infected_by_age_by_time = np.zeros((T, n_ages, n_sector))
    recovered_by_age_by_time = np.zeros((T, n_ages, n_sector))
    dead_by_age_by_time = np.zeros((T, n_ages, n_sector))
    
    
    mild_sector_time = np.zeros((T, n_sector, 3))
    severe_sector_time = np.zeros((T, n_sector, 3))
    critical_sector_time = np.zeros((T, n_sector, 3))
    isolated_sector_time = np.zeros((T, n_sector, 3))
    recovered_sector_time = np.zeros((T, n_sector, 3))
    dead_sector_time = np.zeros((T, n_sector, 3))


    for i in range(initial_infected.shape[0]):
        E[0, initial_infected[i]] = True
        S[0, initial_infected[i]] = False
        time_exposed[initial_infected[i]] = 0
        num_infected_by[initial_infected[i]] = 0
        num_infected_by_outside[initial_infected[i]] = 0
        num_infected_asympt[initial_infected[i]] = 0
        time_to_activation[initial_infected[i]] = functions.threshold_log_normal(time_to_activation_mean, time_to_activation_std)

    E_per_time[0] = initial_infected.shape[0]

    print('Initialized finished')
    

    # Save round 0 results results
    t_now = 0
    t = 0
    


    
    blocks_frac = np.linspace(0, n, n_threads+1)
    blocks = np.zeros(n_threads+1, dtype=np.int32)
    for i in range(blocks_frac.shape[0]):
        blocks[i] = np.round(blocks_frac[i])
    time_doing_work = 0
    time_acc = 0
    print('starting simulation loop')
    start_all_sim = time.time()
    for t in range(1, T):
        #pasted in from start of loop
        t_last = (t+1)%2
        t_now = (t)%2
        if t % 10 == 0 or t == 1:
            print(t,"/",T)
            print(S[t_last].sum())

        if t == t_school_lockdown:
            school_contact = np.zeros(school_contact.shape)
            print("locking down school")

        if t == t_lockdown:

            other_contact = other_contact/lockdown_factor
            work_contact = work_contact/lockdown_factor
            customer_contact = customer_contact/lockdown_factor
            if params['close_schools_lockdown']:
                school_contact = np.zeros(school_contact.shape)
            else:
                school_contact = school_contact/lockdown_factor
            print("lockding down everything", lockdown_factor)
            #TODO: modify work contact during lockdown
        if t == t_second_lockdown:

            other_contact = other_contact_init/second_lockdown_factor
            work_contact = work_contact_init/second_lockdown_factor
            customer_contact = customer_contact_init/second_lockdown_factor
            school_contact = np.zeros(school_contact.shape)
            print("lockding down everything a second time")

        if t == t_lockdown_release:
            other_contact = other_contact*lockdown_factor
            work_contact = work_contact*lockdown_factor
            customer_contact = customer_contact*lockdown_factor
            # school_contact = school_contact_init
            print('releasing lockdown')

        if t == t_tracing_start and tracing_enabled:
            contact_tracing = True
        if t == t_stayinghome_start:
            Home = Home_real


            
        #importations
        if t < t_end_imports:
            num_imports = np.random.poisson(mean_total_imports/t_end_imports)
#            import_infected = np.random.choice(n, num_imports, replace=False)
            import_infected = functions.resevoir_sample(n, num_imports)
            for i in range(num_imports):
                E[0, import_infected[i]] = True
                S[0, import_infected[i]] = False
                time_exposed[import_infected[i]] = t
                num_infected_by[import_infected[i]] = 0
                num_infected_by_outside[import_infected[i]] = 0
                num_infected_asympt[import_infected[i]] = 0
                time_to_activation[import_infected[i]] = functions.threshold_log_normal(time_to_activation_mean, time_to_activation_std)




        S[t_now] = S[t_last]
        E[t_now] = E[t_last]
        Mild[t_now] = Mild[t_last]
        Documented[t_now]=Documented[t_last]
        Severe[t_now] = Severe[t_last]
        Critical[t_now] = Critical[t_last]
        R[t_now] = R[t_last]
        D[t_now] = D[t_last]
        Q[t_now] = Q[t_last]
    
        
        curr_threads = []
        start_time = time.time()
        for thread_idx in range(n_threads):
            
            args = (seed*(n_threads*T) + thread_idx*T + t, t, t_now, t_last, T, blocks[thread_idx], blocks[thread_idx+1], households, age, age_groups, diabetes, hypertension,\
                     other_contact, work_contact, customer_contact, school_contact, sector, edu_sector, customer_facing, age_groups_sector,\
                     p_mild_severe, p_severe_critical, p_critical_death, mean_time_to_isolate_factor, \
                     lockdown_factor_age, fraction_stay_home, params, \
                     S, Mild, Severe, Critical, E, R, D, Q, Documented, Home, traced, time_documented, time_exposed, time_infected, time_critical, time_severe,\
                     time_to_activation, time_to_severe, time_to_critical, time_to_isolate, time_to_death, time_to_recovery, \
                     infected_by, num_infected_by, num_infected_asympt, num_infected_by_outside, \
                     dead_by_age_by_time, infected_by_age_by_time, Work, pinf_mult, Work_status)
#            run_block(*args)
            thread = threading.Thread(target=run_block, args=args)
            thread.start()
            curr_threads.append(thread)
            
        for thread in curr_threads:
            thread.join()
#        print('end simulating')
        end_time = time.time()
        time_doing_work += end_time - start_time
        
        start_acc = time.time()
        q = queue.Queue()
        for thread_idx in range(n_threads):
            args = (q, S[t_now], E[t_now], Mild[t_now], Severe[t_now], Critical[t_now], R[t_now], D[t_now], Q[t_now], Work_status, age, n_ages, sector, n_sector, blocks[thread_idx], blocks[thread_idx+1])
            thread = threading.Thread(target = accumulate_into_queue, args=args)
            thread.start()
            curr_threads.append(thread)
        for thread in curr_threads:
            thread.join()
        
        for thread_idx in range(n_threads):
            s_total, e_total, mild_total, severe_total, critical_total, r_total, d_total, q_total, symptomatic_age, recovered_age, dead_age, mild_sector, severe_sector, critical_sector, isolated_sector, recovered_sector, dead_sector = q.get()
            S_per_time[t] += s_total
            E_per_time[t] += e_total
            Mild_per_time[t] += mild_total
            Severe_per_time[t] += severe_total
            Critical_per_time[t] += critical_total
            R_per_time[t] += r_total
            D_per_time[t] += d_total
            Q_per_time[t] += q_total
            infected_by_age_by_time[t] += symptomatic_age
            recovered_by_age_by_time[t] += recovered_age
            dead_by_age_by_time[t] += dead_age
            
            
            mild_sector_time[t] += mild_sector
            severe_sector_time[t] += severe_sector
            critical_sector_time[t] += critical_sector
            isolated_sector_time[t] += isolated_sector 
            recovered_sector_time[t] += recovered_sector 
            dead_sector_time[t] += dead_sector
            
        end_acc = time.time()
        time_acc += end_acc - start_acc
        



    R_last = R[(T-1)%2]
    D_last = D[(T-1)%2]
    end_all = time.time()
    print('total work', time_doing_work)
    print('total sim', end_all - start_all_sim)
    print('total acc', time_acc)
    print('total', end_all - start_all)
    return S_per_time, E_per_time, Mild_per_time, Documented, Severe_per_time, Critical_per_time,\
      R_per_time, D_per_time, Q_per_time, num_infected_by, time_documented, time_to_activation,\
      time_to_death, time_to_recovery, time_critical, time_exposed, num_infected_asympt, age, \
      time_infected, time_to_severe, R_last, D_last, infected_by_age_by_time, dead_by_age_by_time, recovered_by_age_by_time,\
      mild_sector_time, severe_sector_time, critical_sector_time, isolated_sector_time, recovered_sector_time, dead_sector_time



@jit(nopython=True,nogil=True)
def run_block(seed, t, t_now, t_last, T, start_block, end_block, households, age, age_groups, diabetes, hypertension,\
 other_contact, work_contact, customer_contact, school_contact, sector, edu_sector, customer_facing, age_groups_sector,\
 p_mild_severe, p_severe_critical, p_critical_death, mean_time_to_isolate_factor, \
 lockdown_factor_age, fraction_stay_home, params, \
 S, Mild, Severe, Critical, E, R, D, Q, Documented, Home, traced, time_documented, time_exposed, time_infected, time_critical, time_severe,\
 time_to_activation, time_to_severe, time_to_critical, time_to_isolate, time_to_death, time_to_recovery, \
 infected_by, num_infected_by, num_infected_asympt, num_infected_by_outside, \
 dead_by_age_by_time, infected_by_age_by_time, Work, pinf_mult, Work_status):
    

    np.random.seed(seed)

    time_to_activation_mean = params['time_to_activation_mean']
    time_to_activation_std = params['time_to_activation_std']
    mean_time_to_death = params['mean_time_to_death']
    mean_time_critical_recovery = params['mean_time_critical_recovery']
    mean_time_severe_recovery = params['mean_time_severe_recovery']
    mean_time_to_severe = params['mean_time_to_severe']
    mean_time_mild_recovery = params['mean_time_mild_recovery']
    mean_time_to_critical = params['mean_time_to_critical']
    p_documented_in_mild = params['p_documented_in_mild']
    mean_time_to_isolate_asympt = params['mean_time_to_isolate_asympt']
    asymptomatic_transmissibility = params['asymptomatic_transmissibility']
    p_infect_given_contact = params['p_infect_given_contact']
    mean_time_to_isolate = params['mean_time_to_isolate']
    n_ages = int(params['n_ages'])
    contact_tracing = bool(params['contact_tracing'])
    p_trace_outside = params['p_trace_outside']
    p_trace_household = params['p_trace_household']
    p_infect_household = params['p_infect_household']
    work_gamma_shape = params['gamma_shape_nc']
    work_gamma_scale = params['gamma_scale_nc']
    customer_gamma_shape = params['gamma_shape_c']
    customer_gamma_scale = params['gamma_scale_c']
    p_work_contacts = params['p_work_contacts']
    
    max_household_size = households.shape[1]



    for i in range(start_block, end_block):
        #exposed -> (mildly) infected
        if E[t_last, i]:
            if t - time_exposed[i] == time_to_activation[i]:
                Mild[t_now, i] = True
                time_infected[i] = t
                E[t_now, i] = False
                #draw whether they will progress to severe illness
                if np.random.rand() < p_mild_severe[age[i], diabetes[i], hypertension[i]]:
                    time_to_severe[i] = functions.threshold_exponential(mean_time_to_severe)
                    time_to_recovery[i] = T+1
                #draw time to recovery
                else:
                    time_to_recovery[i] = functions.threshold_exponential(mean_time_mild_recovery)
                    time_to_severe[i] = T+1
                #draw time to isolation
                time_to_isolate[i] = functions.threshold_exponential(mean_time_to_isolate*get_isolation_factor(age[i], mean_time_to_isolate_factor))
                if time_to_isolate[i] == 0:
                    Q[t_now, i] = True
        #symptomatic individuals
        if (Mild[t_last, i] or Severe[t_last, i] or Critical[t_last, i]):
            #recovery
            if t - time_infected[i] == time_to_recovery[i]:
                R[t_now, i] = True
                Mild[t_now, i] = Severe[t_now, i] = Critical[t_now, i] = Q[t_now, i] = False
                continue
            if Mild[t_last, i] and not Documented[t_last, i]:
                if np.random.rand() < p_documented_in_mild:
                    Documented[t_now, i]=True
                    time_documented[i]=t
                    traced[i] = True
                    if contact_tracing:
                        Q[t_now, i] = True
                        do_contact_tracing(i, infected_by, p_trace_outside, Q, S, t, households, p_trace_household, Documented, time_documented, traced)
            #progression between infection states
            if Mild[t_last, i] and t - time_infected[i] == time_to_severe[i]:
                Mild[t_now, i] = False
                Severe[t_now, i] = True
                if not Documented[t_last, i]:
                    Documented[t_now, i]=True
                    time_documented[i]=t
                    traced[i] = True
                    if contact_tracing:
                        Q[t_now, i] = True
                        do_contact_tracing(i, infected_by, p_trace_outside, Q, S, t, households, p_trace_household, Documented, time_documented, traced)
                Q[t_now, i] = True
                time_severe[i] = t
                if np.random.rand() < p_severe_critical[age[i], diabetes[i], hypertension[i]]:
                    time_to_critical[i] = functions.threshold_exponential(mean_time_to_critical)
                    time_to_recovery[i] = T+1
                else:
                    time_to_recovery[i] = functions.threshold_exponential(mean_time_severe_recovery) + time_to_severe[i]
                    time_to_critical[i] = T+1
            elif Severe[t_last, i] and t - time_severe[i] == time_to_critical[i]:
                Severe[t_now, i] = False
                Critical[t_now, i] = True
                time_critical[i] = t
                if np.random.rand() < p_critical_death[age[i], diabetes[i], hypertension[i]]:
                    time_to_death[i] = functions.threshold_exponential(mean_time_to_death)
                    time_to_recovery[i] = T+1
                else:
                    time_to_recovery[i] = functions.threshold_exponential(mean_time_critical_recovery) + time_to_severe[i] + time_to_critical[i]
                    time_to_death[i] = T+1
            #risk of mortality for critically ill patients
            elif Critical[t_last, i]:
                if t - time_critical[i] == time_to_death[i]:
                    Critical[t_now, i] = False
                    Q[t_now, i] = False
                    D[t_now, i] = True
                    
        if E[t_last, i] or Mild[t_last, i] or Severe[t_last, i] or Critical[t_last, i]:
            #not isolated: either enter isolation or infect others
            if not Q[t_last, i]:
                #isolation
                if not E[t_last, i] and t - time_infected[i] == time_to_isolate[i]:
                    Q[t_now, i] = True
                    continue
                if E[t_last, i] and t - time_exposed[i] == time_to_isolate[i]:
                    Q[t_now, i] = True
                    continue
                #infect within family
                for j in range(max_household_size):
                    if households[i,j] == -1:
                        break
                    contact = households[i,j]
                    infectiousness = p_infect_household
                    if E[t_last, i]:
                        infectiousness *= asymptomatic_transmissibility
                    if S[t_last, contact] and np.random.rand() < infectiousness:
                            E[t_now, contact] = True
                            # print('infected in home')

                            
                            num_infected_by[contact] = 0
                            num_infected_by_outside[contact] = 0
                            num_infected_asympt[contact] = 0
                            S[t_now, contact] = False
                            time_to_isolate[contact] = functions.threshold_exponential(mean_time_to_isolate_asympt*get_isolation_factor(age[contact], mean_time_to_isolate_factor))
                            if time_to_isolate[contact] == 0:
                                Q[t_now, contact] = True
                            time_exposed[contact] = t
                            time_to_activation[contact] = functions.threshold_log_normal(time_to_activation_mean, time_to_activation_std)
                            num_infected_by[i] += 1
                            if E[t_last, i]:
                                num_infected_asympt[i] += 1
                #infect across families
                if not Home[i]:
#                    print('start infecting')
                    #work and customer contacts are negative binomial distributed -- use the Gamma-Poisson
                    #mixture representation and the fact that we can split the Poisson distribution over age groups
                    #sector 0 = unemployed, edu_sector is part of the population file and gives the index for primary/secondary education
                    if sector[i] == 0 or sector[i] == edu_sector or Work_status[i] != 0 or np.random.rand() < 1 - p_work_contacts:
                        total_work_contacts = 0
                        total_customer_contacts = 0
                    else:
                        total_work_contacts = np.random.gamma(work_gamma_shape, work_gamma_scale)
                        if customer_facing[i]:
                            total_customer_contacts = np.random.gamma(customer_gamma_shape, customer_gamma_scale)
                        else:
                            total_customer_contacts = 0
#                    print('done total contacts')
                    for contact_age in range(n_ages):
                        infectiousness = p_infect_given_contact*pinf_mult[contact_age]
                        if E[t_last, i]:
                            infectiousness *= asymptomatic_transmissibility

#                        print('start poisson')
                        if age_groups[contact_age].shape[0] == 0:
                            continue
                        #general community contacts
                        num_other_contacts = np.random.poisson(other_contact[age[i], contact_age])
                        #contacts in school -- only children and teachers for now
                        if school_contact[age[i], contact_age] > 0:
                            num_school_contacts = np.random.poisson(school_contact[age[i], contact_age])
                        else:
                            num_school_contacts = 0
                        #coworker contacts
                        if work_contact[sector[i], age[i], contact_age] > 0 and Work_status[i] == 0:
                            num_work_contacts = np.random.poisson(total_work_contacts*work_contact[sector[i], age[i], contact_age])
#                        print('done poisson')
                        else:
                            num_work_contacts = 0
                        #customer contacts
                        if customer_facing[i] and Work_status[i] == 0:
                            num_customer_contacts = np.random.poisson(total_customer_contacts*customer_contact[sector[i], age[i], contact_age])
                        else:
                            num_customer_contacts = 0
                        #infect all of the contacts
#                        print('start infecting', num_other_contacts, num_school_contacts, num_work_contacts, num_customer_contacts)
                        for j in range(num_other_contacts):
                            if np.random.rand() < infectiousness:
#                                print('other infection')
                                contact = np.random.choice(age_groups[contact_age])
                                if S[t_last, contact] and not Home[contact]:
                                    infect_contact(contact, E, Q, t, i, t_last, t_now, num_infected_by, num_infected_by_outside, num_infected_asympt, S, time_to_isolate, mean_time_to_isolate_asympt, age, mean_time_to_isolate_factor, time_exposed, time_to_activation, time_to_activation_mean, time_to_activation_std, contact_tracing, infected_by)
                        for j in range(num_school_contacts):
                            if np.random.rand() < infectiousness:
#                                print('school infection')
                                # if age_groups_sector == 0, then work sectors are not implemented, so sample from general pop
                                if contact_age <= 18 or age_groups_sector[0].shape[0] == 0:
                                    contact = np.random.choice(age_groups[contact_age])
                                    if S[t_last, contact] and not Home[contact]:
                                        infect_contact(contact, E, Q, t, i, t_last, t_now, num_infected_by, num_infected_by_outside, num_infected_asympt, S, time_to_isolate, mean_time_to_isolate_asympt, age, mean_time_to_isolate_factor, time_exposed, time_to_activation, time_to_activation_mean, time_to_activation_std, contact_tracing, infected_by)
                                else:
#                                    print("sampling from teachers")
                                    if age_groups_sector[edu_sector*n_ages + contact_age].shape[0] > 0:
                                        contact = np.random.choice(age_groups_sector[edu_sector*n_ages + contact_age])
                                        if S[t_last, contact] and not Home[contact]:
                                            infect_contact(contact, E, Q, t, i, t_last, t_now, num_infected_by, num_infected_by_outside, num_infected_asympt, S, time_to_isolate, mean_time_to_isolate_asympt, age, mean_time_to_isolate_factor, time_exposed, time_to_activation, time_to_activation_mean, time_to_activation_std, contact_tracing, infected_by)
                        for j in range(num_work_contacts):
                            if np.random.rand() < infectiousness:
#                                print('work infection')
#                                print(len(age_groups_sector), sector[i], n_ages, contact_age)
                                #age_group_sector is a list of numpy arrays. Each array lists the agents in a given sector and age group. The list is flattened
                                #because of numba, so it is indexed as a 1-d list where the index combines the sector and age group indices
                                if age_groups_sector[sector[i]*n_ages + contact_age].shape[0] > 0:
                                    contact = np.random.choice(age_groups_sector[sector[i]*n_ages + contact_age])
                                    #continue drawing until we get someone who is not WFH or laid off
                                    while Work_status[contact] != 0:
                                        contact = np.random.choice(age_groups_sector[sector[i]*n_ages + contact_age])
                                    if S[t_last, contact] and not Home[contact]:
                                        infect_contact(contact, E, Q, t, i, t_last, t_now, num_infected_by, num_infected_by_outside, num_infected_asympt, S, time_to_isolate, mean_time_to_isolate_asympt, age, mean_time_to_isolate_factor, time_exposed, time_to_activation, time_to_activation_mean, time_to_activation_std, contact_tracing, infected_by)
                        for j in range(num_customer_contacts):
                            if np.random.rand() < infectiousness:
#                                print('customer infection')
                                contact = np.random.choice(age_groups[contact_age])
                                if S[t_last, contact] and not Home[contact]:
                                    infect_contact(contact, E, Q, t, i, t_last, t_now, num_infected_by, num_infected_by_outside, num_infected_asympt, S, time_to_isolate, mean_time_to_isolate_asympt, age, mean_time_to_isolate_factor, time_exposed, time_to_activation, time_to_activation_mean, time_to_activation_std, contact_tracing, infected_by)
#                        print('done infecting')

@jit(nopython=True,nogil=True)                          
def infect_contact(contact, E, Q, t, i, t_last, t_now, num_infected_by, num_infected_by_outside, num_infected_asympt, S, time_to_isolate, mean_time_to_isolate_asympt, age, mean_time_to_isolate_factor, time_exposed, time_to_activation, time_to_activation_mean, time_to_activation_std, contact_tracing, infected_by):                            
    E[t_now, contact] = True



    num_infected_by[contact] = 0
    num_infected_by_outside[contact] = 0
    num_infected_asympt[contact] = 0
    S[t_now, contact] = False
    time_to_isolate[contact] = functions.threshold_exponential(mean_time_to_isolate_asympt*get_isolation_factor(age[contact], mean_time_to_isolate_factor))
    if time_to_isolate[contact] == 0:
        Q[t_now, contact] = True
    time_exposed[contact] = t
    time_to_activation[contact] = functions.threshold_log_normal(time_to_activation_mean, time_to_activation_std)
    num_infected_by[i] += 1
    if contact_tracing:
        infected_by[i, num_infected_by_outside[i]] = contact
    num_infected_by_outside[i] += 1
    if E[t_last, i]:
        num_infected_asympt[i] += 1


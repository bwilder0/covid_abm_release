import numpy as np
import scipy.stats
#import sample_households
#import sample_comorbidities

def calibrate_p_document_mild(p_target, country, p_mild_severe, mean_time_mild_recovery, mean_time_to_severe):
    n  = 10000
#    if country == "Italy":
#      households, age = sample_households.sample_households_italy(n)      
#    elif country == "Germany":
#      households, age = sample_households.sample_households_germany(n)
#    elif country == "UK":
#      households, age = sample_households.sample_households_uk(n)
#    elif country == "Spain":
#      households, age = sample_households.sample_households_spain(n)
#    elif country == "China": 
#      households, age = sample_households.sample_households_china(n)
#    else:
#      households, age = sample_households.sample_households_un(n, country)
#    
#    diabetes, hypertension = sample_comorbidities.sample_joint_comorbidities(age, country)
   
    def total_p_document(p_document):
#        time_document = np.random.geometric(p_document, size=n)
#        recovery = np.random.rand(n) < p_mild_severe[age, diabetes, hypertension]
#        time_to_recovery = np.random.exponential(mean_time_mild_recovery, size=n)
#        time_to_severe = np.random.exponential(mean_time_to_severe, size=n)
#        return (time_document <= recovery*time_to_recovery + (1-recovery)*time_to_severe).mean()
        time_document = np.random.geometric(p_document, size=n)
        time_to_recovery = np.random.exponential(mean_time_mild_recovery, size=n)
        return (time_document < time_to_recovery).mean()

    
    eps = 0.0001
    ub = 1
    lb = 0
    p_document = (lb + ub)/2.
    cumulative_sar = total_p_document(p_document)
    while np.abs(cumulative_sar - p_target) > eps:
        if cumulative_sar < p_target:
            lb = p_document
            p_document = (p_document + ub)/2
        else:
            ub = p_document
            p_document = (p_document + lb)/2
        cumulative_sar = total_p_document(p_document)
    return p_document

    
def get_p_infect_household(mean_time_to_isolate, time_to_activation_mean, time_to_activation_std, asymptomatic_transmissibility):
#    age_groups = tuple([np.where(age == i)[0] for i in range(0, n_ages)])
#    age_group_sizes = np.array([age_groups[i].shape[0] for i in range(0, n_ages)])
#    #uniform distribution over contact probabilities for now
#    p = age_group_sizes/age_group_sizes.sum()
#    p = p.reshape((1, p.shape[0]))
#    beta = p.repeat(n_ages, axis=0)
        
    
    #probability of an infected patient to recover each day
    #https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf 
    #suggests "the median time from onset to clinical recovery for mild
    #cases is approximately 2 weeks and is 3-6 weeks for patients with severe or critical disease. 
    #setting this to a fairly arbitrary average of 16 days to recovery
#    p_recover = 1./16
    
    #cumulative probability of death
    #from http://weekly.chinacdc.cn/en/article/id/e53946e2-c6c4-41e9-9a9b-fea8db1a8f51
    #better sources?
#    p_death_target = np.zeros(n_ages)
#    p_death_target[:10] = 0
#    p_death_target[10:20] = 0.2
#    p_death_target[20:30] = 0.2
#    p_death_target[30:40] = 0.2
#    p_death_target[40:50] = 0.4
#    p_death_target[50:60] = 1.3
#    p_death_target[60:70] = 3.6
#    p_death_target[70:80] = 8.0
#    p_death_target[80:] = 14.8
#    p_death_target /= 100
    
    #set the per-day mortality rate to match the overall CFR
    
    #probability bernouli with probability p1 succeeds before bernouli with probability p2
    def p_event_before_another(p1, p2):
        p = 0
        #probability of event 1 on day k and no event 2 before day k
        for k in range(1, 100):
            p += (1 - p1)**(k-1) * p1 * (1 - p2)**k
        return p
    
    def bernoulli_before_exponential(p_bern, lam):
        p = 0
        #probability of event 1 on day k and no event 2 before day k
        for k in range(1, 100):
            p += (1 - p_bern)**(k-1) * p_bern * np.exp(-k/lam)
        return p


    def bernoulli_before_exp_lognorm(p_bern, lam, logmean, logstd):
        p = 0
        #probability of event 1 on day k and no exponential/lognormal events before day k
        for k in range(1, 100):
            p += (1 - p_bern)**(k-1) * p_bern * np.exp(-k/lam) * (1 - scipy.stats.lognorm.cdf(k, s = logstd, scale=np.exp(logmean)))
        return p

    def bernoulli_before_exp_lognorm(p_bern, lam, logmean, logstd):
        p = 0
        #probability of event 1 on day k and no exponential/lognormal events before day k
        for k in range(1, 100):
            p += (1 - p_bern)**(k-1) * p_bern * np.exp(-k/lam) * (1 - scipy.stats.lognorm.cdf(k, s = logstd, scale=np.exp(logmean)))
        return p
    
    
    def threshold_exponential(mean, num):
        return 1 + np.round(np.random.exponential(mean-1, size=num))
    
    def threshold_log_normal(mean, sigma, num):
        x = np.random.lognormal(mean, sigma, size=num)
        x[x <= 0] = 1
        return np.round(x)

    def total_p_infection(p_infect, mean_time_to_isolate, time_to_activation_mean, time_to_activation_std):
        time_to_isolate = threshold_exponential(mean_time_to_isolate, 5000)
        time_to_activate = threshold_log_normal(time_to_activation_mean, time_to_activation_std, 5000)
        time_infect_asymp = np.random.geometric(p_infect*asymptomatic_transmissibility, size=5000)
        time_infect_symp = np.random.geometric(p_infect, size=5000)
        return (1 - (time_infect_asymp > time_to_activate)*(time_infect_symp > time_to_isolate)).mean()
    


#    
#    p_death = p_death_target/5
#    ubs = np.ones(n_ages)
#    lbs = np.zeros(n_ages)
    eps = 0.0001
#    #binary search for the parameter for each age group
#    for i in range(10, n_ages):
#        cumulative_prob_death = p_event_before_another(p_death[i], p_recover) 
#        while np.abs(cumulative_prob_death - p_death_target[i]) > eps:
#            if cumulative_prob_death < p_death_target[i]:
#                lbs[i] = p_death[i]
#                p_death[i] = (p_death[i] + ubs[i])/2
#            else:
#                ubs[i] = p_death[i]
#                p_death[i] = (p_death[i] + lbs[i])/2
#            p_death[i] = (ubs[i] + lbs[i])/2
#            cumulative_prob_death = p_event_before_another(p_death[i], p_recover) 
    
    #probability for an infected patient to infect each member of the household each day
    #calibrate to match
    #https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30462-1/fulltext
    target_secondary_attack_rate = 0.35
    ub = 1
    lb = 0
    p_infect_household = 0.5
    cumulative_sar = total_p_infection(p_infect_household, mean_time_to_isolate, time_to_activation_mean, time_to_activation_std)
    while np.abs(cumulative_sar - target_secondary_attack_rate) > eps:
        if cumulative_sar < target_secondary_attack_rate:
            lb = p_infect_household
            p_infect_household = (p_infect_household + ub)/2
        else:
            ub = p_infect_household
            p_infect_household = (p_infect_household + lb)/2
        cumulative_sar = total_p_infection(p_infect_household, mean_time_to_isolate, time_to_activation_mean, time_to_activation_std)
        
    return p_infect_household
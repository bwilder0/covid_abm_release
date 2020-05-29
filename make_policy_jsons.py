
country = 'lombardy'
frac = 1.0
for distance in [True, False]:
    if distance:
        distance_str = '_distance'
        lockdown_year = 2
    else:
        distance_str = ''
        lockdown_year = 3
    for index in range(4):
        frac_stay_home = [0]*4
        frac_stay_home[index] = frac
        command_nyc = \
'''{{
	"country":"NYC",
	"bayesian":"True",
	"p_infect_given_contact_range": [0.03, 0.07],
	"mortality_multiplier_range": [1, 4],
	"dmult_mult":1.0,
	"start_date_list": ["02-10-2020"],
	"start_infected_range": [5, 500],
	"pinf_mult_range": [2.5],
	"lockdown_factor":2.0,
	"second_lockdown_factor":10.0,

	"d_school_lockdown":"12-12-3000",
	"d_lockdown":"03-16-20{3}0",
	"d_second_lockdown":"03-23-2030",
	"d_stay_home":"03-23-2020",

	"d_lockdown_release":"12-12-3000",

	"d_end":"06-29-2020",
	"close_schools_lockdown":"False",

	"stay_home_groups":[0, 20, 40, 60, 100],
	"fraction_stay_home":{0},
	
	"load_posterior":"True",
	"posterior_file":"NYC_posterior.pickle",

	"runs_to_combine":["nyc_bayesian_policy_{1}_{2}{4}"],
	"jobs_per_combo_per_batch":[1],
	"combined_dir":"nyc_bayesian_policy_{1}_{2}{4}_combined"
}}'''.format(frac_stay_home, frac, index, lockdown_year, distance_str)

        command_hubei =\
'''{{
	"country":"China",
	"bayesian":"True",
	"p_infect_given_contact_range": [0.020, 0.035],
	"mortality_multiplier_range": [1, 3],
	"start_date_list": ["11-10-2019", "11-11-2019", "11-12-2019", "11-13-2019", "11-14-2019", "11-15-2019", "11-16-2019", "11-17-2019", "11-18-2019", "11-19-2019", "11-20-2019"],
	"dmult_mult":1.0,
	"pinf_mult_range": [1.25],
	"lockdown_factor":2.0,

	"d_school_lockdown":"12-12-3000",
	"d_lockdown":"01-23-20{3}0",
	"d_stay_home":"01-23-2020",

	"d_lockdown_release":"12-12-3000",

	"d_end":"06-29-2020",
	"close_schools_lockdown":"False",

	"stay_home_groups":[0, 20, 40, 60, 100],
	"fraction_stay_home":{0},
	
	"load_posterior":"True",
	"posterior_file":"China_posterior.pickle",

	"runs_to_combine":["hubei_bayesian_policy_{1}_{2}{4}"],
	"jobs_per_combo_per_batch":[1],
	"combined_dir":"hubei_bayesian_policy_{1}_{2}{4}_combined"
}}'''.format(frac_stay_home, frac, index, lockdown_year, distance_str)

        command_lombardy =\
'''{{
	"country":"Italy",
	"bayesian":"True",
	"p_infect_given_contact_range": [0.025, 0.04],
	"mortality_multiplier_range": [1],
	"dmult_mult":4.0,
	"start_date_list": ["01-15-2020", "01-16-2020", "01-17-2020", "01-18-2020", "01-19-2020", "01-20-2020", "01-21-2020", "01-22-2020", "01-23-2020", "01-24-2020", "01-25-2020"],
	"pinf_mult_range": [1.25],
	"lockdown_factor":2.0,

	"d_school_lockdown":"12-12-3000",
	"d_lockdown":"03-08-2020",
	"d_lockdown_release":"12-12-3000",
	"d_stay_home":"03-08-2020",

	"d_lockdown_release":"12-12-3000",

	"d_end":"06-29-2020",
	"close_schools_lockdown":"False",
	
	"stay_home_groups":[0, 20, 40, 60, 100],
	"fraction_stay_home":{0},
	
	"load_posterior":"True",
	"posterior_file":"Italy_posterior.pickle",

	"runs_to_combine":["lombardy_bayesian_policy_{1}_{2}{4}"],
	"jobs_per_combo_per_batch":[1],
	"combined_dir":"lombardy_bayesian_policy_{1}_{2}{4}_combined"
}}'''.format(frac_stay_home, frac, index, lockdown_year, distance_str)
        if country == 'hubei':
            command = command_hubei
        elif country == 'nyc':
            command = command_nyc
        elif country == 'lombardy':
            command = command_lombardy
        else:
            raise Exception('unknown country')
        with open('inputs/{3}_bayesian_policy_{0}_{1}{2}.json'.format(frac, index, distance_str, country), 'w') as f:
            f.write(command)
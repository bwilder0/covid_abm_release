import subprocess
import argparse

parser = argparse.ArgumentParser(description='Combine results from many runs.')

parser.add_argument('--country', type=str, default = 'lombardy', help='JSON file of input parameters')
parser.add_argument('--frac', type=float, default = 0.5, help='JSON file of input parameters')

args = parser.parse_args()

#country = 'hubei'
country = args.country
frac = args.frac
for distance in [True, False]:
    if distance:
        distance_str = '_distance'
        lockdown_year = 2
    else:
        distance_str = ''
        lockdown_year = 3
    for index in range(4):

        command = "sbatch --array=0-100 job.parameter_sweep_{}_bayesian_both.sh 10 10 {}_bayesian_policy_{}_{}{} 0".format(country, country, frac, index, distance_str)
        print(command)
        subprocess.call(command, shell=True)
        
        
import json
import pandas as pd
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Combine results from many runs.')


parser.add_argument('--EXP_ROOT_DIR', type=str, default='../../parameter_sweeps/npi_sectors', help="On fasrc, you should specify a directory\
                                            that you've created in long term file storage, e.g., \
                                            /n/holylfs/LABS/tambe_lab/jkillian/covid19/parameter_sweeps \
                                            \nIMPORTANT: make sure you have already created this directory (this script will assume it exists.)")

parser.add_argument('--country', type=str, default = 'lombardy', help='JSON file of input parameters')
parser.add_argument('--njobs', type=int, default = 0, help='number of jobs to merge')
parser.add_argument('--n', type=int, default = 1, help='number of processes')
parser.add_argument('--frac', type=float, default = 0.5, help='number of processes')


args = parser.parse_args()

EXP_DIR = args.EXP_ROOT_DIR

country = args.country
input_file = 'inputs/{}_bayesian_policy_{}_{}{}.json'
for index_group in range(4):
    for distancing_str in ['', '_distance']:
        
        infile = input_file.format(country, args.frac, index_group, distancing_str)
        print(infile)
        input_dict = json.load(open(infile, 'r'))
        N = float(10e6)
        if input_dict['country'] == 'NYC':
            N = float(8.4e6)
        run_name = input_dict['runs_to_combine'][0]
        dirname = os.path.join(EXP_DIR, run_name)
        combined_dir = input_dict['combined_dir']
        dirname = os.path.join(EXP_DIR, combined_dir)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
        path = os.path.join(EXP_DIR, run_name)
        
            
        datatypes_n_t = [
            'susceptible', 
            'exposed', 
            'deaths', 
            'mild', 
            'severe', 
            'critical', 
            'recovered', 
            'quarantine',     
            ]
        datatypes_n_1 = [
                'infected_start_store',
                'r0_tot', 
                'mse',
                'ifr_tot',
                'frac_death_older',
                'pinf_mult_vals_store',
                'sd_vals_store',
                'mm_vals_store',
                'pigc_vals_store'
        ]
        
        
        out_template = combined_dir+'_bayesian_n%s_i%s_%s.hdf'
        out_template = os.path.join(dirname, out_template)
        
        
        def load_thing(fname):
            try:
                a = pd.read_hdf(fname).to_numpy()
            except:
                print('File not found: ', fname)
                a = None
            return a
        
        import multiprocessing
        pool = multiprocessing.Pool(args.n)
        
        for d in datatypes_n_1 + datatypes_n_t:
            print(d)
        #    all_results = []
            all_files = []
            for index in range(args.njobs):
                fname = '%s_bayesian_n%s_i%s'%(run_name, N, index)
                fname += '_%s.hdf'
                fname = fname%d
                fname = os.path.join(path, fname)
                all_files.append(fname)
            all_results = pool.map(load_thing, all_files)
            all_results = [x for x in all_results if x is not None]
            
            combined = np.vstack(all_results)
            df = pd.DataFrame(combined)
            outfile = out_template%(N, 0, d)
            df.to_hdf(outfile, key='values')
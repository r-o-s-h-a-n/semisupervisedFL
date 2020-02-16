# import objgraph
# from guppy3 import hpy
import argparse
import os
# import gc
import imp
import datetime
import numpy as np
import experiments as exp
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, default='', help='config file with parameters of the experiment')
parser.add_argument('--run', type=int, required=False, default=0, help='run number for current experiment')
args_opt = parser.parse_args()

curr_run_number = int(args_opt.run)
# sets unique directory name for writing all logs for experiment and saved models
log_directory = os.path.join('.', 'logs', args_opt.exp)

# loads experiment
exp_config_file = os.path.join('.','config',args_opt.exp+'.py')
# loads parameter handler
ph = imp.load_source("",exp_config_file).ph
ph['log_dir'] = log_directory # set log directory
ph._init_hparams()

if ph['sample_client_data']:
        print('\n\n WARNING: training on a sample of 100 clients with max 100 examples each.\
                If this is not intended behavior, please set `sample_client_data` to \
                False in the config file\n')

ph.set_hparams(ph.hparam_sets[curr_run_number], curr_run_number)
experiment = getattr(exp, ph['experiment'])(ph)
experiment.run()
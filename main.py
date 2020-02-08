import argparse
import os
import imp
import datetime
import numpy as np
import experiments as exp
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, default='', help='config file with parameters of the experiment')
args_opt = parser.parse_args()

# sets unique directory name for writing all logs for experiment and saved models
log_directory = os.path.join('.',
                            'logs',
                            '_'.join([args_opt.exp,
                                    datetime.datetime.now().isoformat()
                            ]))

# loads experiment
exp_config_file = os.path.join('.','config',args_opt.exp+'.py')

# loads parameter handler
ph = imp.load_source("",exp_config_file).ph
ph['log_dir'] = log_directory # set log directory
ph._init_hparams()

# iterate through cartesian product of hyperparameters and run experiment
hparam_sets = list(ph.gen_hparam_cartesian_product())
for i in range(ph['curr_run_number'], len(hparam_sets)):
    ph.set_hparams(hparam_sets[i], i)
    experiment = getattr(exp, ph['experiment'])(ph)
    experiment.run()
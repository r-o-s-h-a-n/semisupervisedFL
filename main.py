import argparse
import os
import imp
import experiments as exp
import dataloader as dta


parser = argparse.ArgumentParser()
parser.add_argument('--exp',         type=str, required=True, default='',  help='config file with parameters of the experiment')
args_opt = parser.parse_args()

exp_directory = os.path.join('.','experiments',args_opt.exp)

exp_config_file = os.path.join('.','config',args_opt.exp+'.py')
config = imp.load_source("",exp_config_file).config
config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored

dataset_train, dataset_test = dta.get_client_data(config['dataset_name'],
                                                config['mask_by'],
                                                config['mask_ratios'],
                                                config['sample_client_data']
                                                )

experiment = getattr(exp, config['experiment'])(config)

# TODO add saving and loading from checkpoint

experiment.solve(dataset_train, dataset_test)
experiment.evaluate(dataset_test)


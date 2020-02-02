import argparse
import os
import imp
import experiments as exp
import dataloader as dta


parser = argparse.ArgumentParser()
parser.add_argument('--exp',         type=str, required=True, default='',  help='config file with parameters of the experiment')
args_opt = parser.parse_args()

log_directory = os.path.join('.','logs',args_opt.exp)

exp_config_file = os.path.join('.','config',args_opt.exp+'.py')
config = imp.load_source("",exp_config_file).config
config['log_dir'] = log_directory # the place where logs, models, and other stuff will be stored

train_client_data, test_dataset = dta.get_client_data(**config['dataset'])

experiment = getattr(exp, config['experiment'])(config)

# TODO add saving and loading from checkpoint

# print('\nTraining model\n\n')
experiment.solve(train_client_data)

# print('\nEvaluating model\n\n')
experiment.evaluate(test_dataset)



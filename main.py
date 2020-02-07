import argparse
import os
import imp
import numpy as np
import experiments as exp
import dataloader as dta
from utils import gen_hp_cartesian_product
# import tensorflow as tf
# from tensorboard.plugins.hparams import api as hp


parser = argparse.ArgumentParser()
parser.add_argument('--exp',         type=str, required=True, default='',  help='config file with parameters of the experiment')
args_opt = parser.parse_args()

log_directory = os.path.join('.','logs',args_opt.exp)

exp_config_file = os.path.join('.','config',args_opt.exp+'.py')
config = imp.load_source("",exp_config_file).config
hparam_classes = imp.load_source("",exp_config_file).hparam_classes
config['log_dir'] = log_directory # the place where logs, models, and other stuff will be stored


for i, hps in enumerate(gen_hp_cartesian_product(list(hparam_classes.values()))):
    from pprint import pprint
    pprint(hps)

    config['run_number'] = i
    # train_client_data, test_dataset = dta.get_client_data(dataset_name = hps[hparam_classes['dataset']],
    #                                                     mask_by = hps[hparam_classes['mask_by']],
    #                                                     mask_ratios = {'unsupervised': hps[hparam_classes['unsupervised_mask_ratio']],
    #                                                                     'supervised': hps[hparam_classes['supervised_mask_ratio']]
    #                                                     },
    #                                                     sample_client_data = config['sample_client_data']
    # )

    experiment = getattr(exp, config['experiment'])(config, hps, hparam_classes)

    experiment.run()
    

    # run_dir = config['log_dir'] + 'run_{}'.format(str(i))
    # with tf.summary.create_file_writer(run_dir).as_default():
    #     hp.hparams(hps)
    
    #     experiment = getattr(exp, config['experiment'])(config, hps)

    #     # print('\nTraining model\n\n')
    #     experiment.solve(train_client_data)

    #     # print('\nEvaluating model\n\n')
    #     experiment.evaluate(test_dataset)
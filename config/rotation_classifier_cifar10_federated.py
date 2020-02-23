import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from parameter_handler import ParameterHandler


######### GENERAL CONFIG ###############
config = {}

config['experiment'] = 'SupervisedLearningFL'
config['model_fn'] = 'RotationSupervisedModel'
config['sample_client_data'] = True      # must set to False when running real experiments
config['curr_run_number'] = 0                  # always initialize as 0, unless starting from a certain run

# data loading
config['shuffle_buffer'] = 1

# training
config['num_rounds'] = 200
config['num_clients_per_round'] = 10
config['num_epochs'] = 5
config['log_every'] = 1
config['model_fp'] = 'rotation_label_classifier.h5'
# config['pretrained_model_fp'] = 'logs/rotation_feature_cifar10/run_0/rotation_feature.h5' # PLEASE CHANGE ME TO THE LOCATION OF THE PRETRAINED FEATURE EXTRACTOR MODEL
config['pretrained_model_fp'] = None

config['optimizer'] = 'SGD'
config['nesterov'] = True
config['learning_rate'] = 0.004
config['momentum'] = 0.9
config['decay'] = 0 #5E-4

######### EXPERIMENTAL PARAMETERS ###############
hparam_map = {'supervised_mask_ratio': hp.HParam('supervised_mask_ratio', hp.Discrete([0.0])),
                'unsupervised_mask_ratio': hp.HParam('unsupervised_mask_ratio', hp.Discrete([0.0])),
                'mask_by': hp.HParam('mask_by', hp.Discrete(['example'])),
                'dataset': hp.HParam('dataset', hp.Discrete(['cifar100'])),
                'batch_size': hp.HParam('batch_size', hp.Discrete([32])),
}

######### METRICS ###############################
metric_map = {'train_loss': hp.Metric('train_loss', display_name='Train Loss'),
              'train_accuracy': hp.Metric('train_accuracy', display_name='Train Accuracy'),
              'test_loss': hp.Metric('test_loss', display_name='Test Loss'),
              'test_accuracy': hp.Metric('test_accuracy', display_name='Test Accuracy')
}

#################################################
ph = ParameterHandler(config, hparam_map, metric_map)
ph.hparam_sets = list(ph.gen_hparam_cartesian_product())
print(len(ph.hparam_sets))

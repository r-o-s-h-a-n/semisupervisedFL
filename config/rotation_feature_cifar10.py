import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from parameter_handler import ParameterHandler


######### GENERAL CONFIG ###############
config = {}

config['experiment'] = 'SupervisedLearningCentral'
config['model_fn'] = 'RotationSelfSupervisedModel'
config['sample_client_data'] = False      # must set to False when running real experiments
config['curr_run_number'] = 0                  # always initialize as 0, unless starting from a certain run

# data loading
config['shuffle_buffer'] = 100

# training
config['num_epochs'] = 10
config['log_every'] = 1
config['model_fp'] = 'rotation_feature.h5'

config['optimizer'] = 'SGD'
config['learning_rate'] = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[35, 70, 85], 
                                                                                values=[0.1,0.02,0.004,0.0008])
config['nesterov'] = True
config['momentum'] = 0.9
config['decay'] = 5E-4

######### EXPERIMENTAL PARAMETERS ###############
HP_SUPERVISED_MASK_RATIO = hp.HParam('supervised_mask_ratio', hp.Discrete([0.0]))
HP_UNSUPERVISED_MASK_RATIO = hp.HParam('unsupervised_mask_ratio', hp.Discrete([0.0]))
HP_MASK_BY = hp.HParam('mask_by', hp.Discrete(['example']))
HP_DATASET = hp.HParam('dataset', hp.Discrete(['cifar10central']))

######### NN HYPERPARAMETERS ####################
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([128]))


hparam_map = {'supervised_mask_ratio': HP_SUPERVISED_MASK_RATIO,
                'unsupervised_mask_ratio': HP_UNSUPERVISED_MASK_RATIO,
                'mask_by': HP_MASK_BY,
                'dataset': HP_DATASET,
                'batch_size': HP_BATCH_SIZE,
}



######### METRICS ###############################
METRIC_TRAIN_LOSS = hp.Metric('train_loss', display_name='Train Loss')
METRIC_TRAIN_ACCURACY = hp.Metric('train_accuracy', display_name='Train Accuracy')
METRIC_TEST_LOSS = hp.Metric('test_loss', display_name='Test Loss')
METRIC_TEST_ACCURACY = hp.Metric('test_accuracy', display_name='Test Accuracy')


metric_map = {'train_loss': METRIC_TRAIN_LOSS,
              'train_accuracy': METRIC_TRAIN_ACCURACY,
              'test_loss': METRIC_TEST_LOSS,
              'test_accuracy': METRIC_TEST_ACCURACY
}


#################################################
ph = ParameterHandler(config, hparam_map, metric_map)
ph.hparam_sets = list(ph.gen_hparam_cartesian_product())
print(len(ph.hparam_sets))

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from parameter_handler import ParameterHandler


######### GENERAL CONFIG ###############
config = {}

config['experiment'] = 'SupervisedLearningCentral'
config['model_fn'] = 'RotationSupervisedModel'
config['sample_client_data'] = True      # must set to False when running real experiments
config['curr_run_number'] = 0            # always initialize as 0, unless starting from a certain run

# data loading
config['shuffle_buffer'] = 100

# training
config['num_epochs'] = 1
config['log_every'] = 1
config['model_fp'] = 'classifier_{}.h5'

######### EXPERIMENTAL PARAMETERS ###############
HP_SUPERVISED_MASK_RATIO = hp.HParam('supervised_mask_ratio', hp.Discrete([0.0]))
HP_UNSUPERVISED_MASK_RATIO = hp.HParam('unsupervised_mask_ratio', hp.Discrete([0.0]))
# HP_PRETRAINED_MODEL = hp.HParam('pretrained_model', hp.Discrete([None]))
HP_MASK_BY = hp.HParam('mask_by', hp.Discrete(['example']))
HP_DATASET = hp.HParam('dataset', hp.Discrete(['emnist']))

######### NN HYPERPARAMETERS ####################
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.02]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['SGD']))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([256]))

######### FL HYPERPARAMETERS ####################
hparam_map = {'supervised_mask_ratio': HP_SUPERVISED_MASK_RATIO,
                'unsupervised_mask_ratio': HP_UNSUPERVISED_MASK_RATIO,
                # 'pretrained_model': HP_PRETRAINED_MODEL,
                'mask_by': HP_MASK_BY,
                'dataset': HP_DATASET,

                'learning_rate': HP_LEARNING_RATE,
                'optimizer': HP_OPTIMIZER,
                'batch_size': HP_BATCH_SIZE
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

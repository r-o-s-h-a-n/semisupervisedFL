import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


######### GENERAL CONFIG ###############
config = {}

config['experiment'] = 'SupervisedLearning'
config['model_fn'] = 'ClassifierModel'
config['verbose'] = False
config['sample_client_data'] = False # must set to False when running real experiments
config['run_number'] = 0

# data loading
config['preprocess_fn'] = 'preprocess_classifier'
config['shuffle_buffer'] = 100

# training
config['num_rounds'] = 1
config['model_fp'] = 'classifier_{}.h5'
# config['evaluation_fp'] = 'classifier.json'

######### EXPERIMENTAL PARAMETERS ###############
HP_SUPERVISED_MASK_RATIO = hp.HParam('supervised_mask_ratio', hp.Discrete([0.5]))
HP_UNSUPERVISED_MASK_RATIO = hp.HParam('unsupervised_mask_ratio', hp.Discrete([0.0]))
# HP_PRETRAINED_MODEL = hp.HParam('pretrained_model', hp.Discrete([None]))
HP_MASK_BY = hp.HParam('mask_by', hp.Discrete(['example']))
HP_DATASET = hp.HParam('dataset', hp.Discrete(['emnist']))

######### NN HYPERPARAMETERS ###############
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.02]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['SGD']))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))

######### FL HYPERPARAMETERS ###############
HP_NUM_CLIENTS_PER_ROUND = hp.HParam('num_clients_per_round', hp.Discrete([32]))
HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([1,5]))


hparam_classes = {'supervised_mask_ratio': HP_SUPERVISED_MASK_RATIO,
                'unsupervised_mask_ratio': HP_UNSUPERVISED_MASK_RATIO,
                # 'pretrained_model': HP_PRETRAINED_MODEL,
                'mask_by': HP_MASK_BY,
                'dataset': HP_DATASET,

                'learning_rate': HP_LEARNING_RATE,
                'optimizer': HP_OPTIMIZER,
                'batch_size': HP_BATCH_SIZE,

                'num_clients_per_round': HP_NUM_CLIENTS_PER_ROUND,
                'num_epochs': HP_NUM_EPOCHS
}

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/supervised_hp/').as_default():
  hp.hparams_config(
    hparams=hparam_classes.values(),
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
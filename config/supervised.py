import architectures as arc 
import dataloader as dta
config = {}

config['experiment'] = 'SupervisedLearning'
config['dataset_name'] = 'emnist'
config['mask_by'] = 'example'
config['mask_ratios'] = {'supervised':0.2, 'unsupervised':0.0}
config['sample_client_data'] = True
config['shuffle_buffer'] = 100
config['batch_size'] = 128
config['unsupervised'] = True
config['max_client_examples'] = 1000 # randomly select max of this many examples on each client per iteration
config['num_rounds'] = 1
config['num_epochs'] = 1
config['model_fn'] = arc.create_compiled_classifier_keras_model
config['num_clients_per_round'] = 64
config['model_fp'] = 'experiments/classifier.h5'
config['preprocess_fn'] = dta.preprocess_classifier
config['verbose'] = True

model_opt = {}
model_opt['learning_rate'] = 1.0
model_opt['saved_model_fp'] = None

config['model_opt'] = model_opt
config['evaluation_fp'] = 'experiments/classifier.json'

config['results'] = {}

# set the parameters related to the training and testing set
# data_train_opt = {} 
# data_train_opt['dataset_name'] = 'emnist'
# data_train_opt['mask_by'] = 'example'
# data_train_opt['mask_ratios'] = {'supervised':0.2, 'unsupervised':0.0}
# data_train_opt['sample_client_data'] = True
# data_train_opt['shuffle_buffer'] = 500
# data_train_opt['batch_size'] = 128
# data_train_opt['unsupervised'] = True
# data_train_opt['max_client_examples'] = 10000 # randomly select max of this many examples on each client per iteration
# data_train_opt['num_epochs'] = 10

# config['data_train_opt'] = data_train_opt

# TODO add settings for model hyperparameters

# net_opt = {}
# net_opt['num_classes'] = 10
# net_opt['encoder_size] = 256

# networks = {}
# # net_optim_params = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(60, 0.1),(120, 0.02),(160, 0.004),(200, 0.0008)]}
# networks['model'] = {'def_file': 'architectures/NetworkInNetwork.py', 'pretrained': None, 'opt': net_opt,  'optim_params': net_optim_params} 
# config['networks'] = networks

# criterions = {}
# criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
# config['criterions'] = criterions
# config['algorithm_type'] = 'SupervisedModel'
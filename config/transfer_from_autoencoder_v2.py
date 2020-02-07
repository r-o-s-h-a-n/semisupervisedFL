config = {}

config['experiment'] = 'SupervisedLearning'
config['verbose'] = True

dataset = {}
dataset['dataset_name'] = 'emnist'
dataset['mask_by'] = 'example'
dataset['mask_ratios'] = {'supervised':0.0, 
                        'unsupervised':0.0}
dataset['sample_client_data'] = False # must set to False when running real experiments
config['dataset'] = dataset

# data loading
config['preprocess_fn'] = 'preprocess_classifier'
config['shuffle_buffer'] = 100
config['batch_size'] = 64
config['num_epochs'] = 10

# training
config['model_fn'] = 'ClassifierModel'
config['num_rounds'] = 40
config['num_clients_per_round'] = 100
config['learning_rate'] = 0.02

config['saved_model_fp'] = 'logs/autoencoder/autoencoder_US_mask_{}.h5'.format(str(dataset['mask_ratios']['unsupervised']))
# config['model_fp'] = 'classifier_S_mask_{}_transfer_autoencoder_US_mask_{}.h5'.format(str(dataset['mask_ratios']['supervised']), 
#                                                                                         str(dataset['mask_ratios']['unsupervised']))
# config['evaluation_fp'] = 'classifier_S_mask_{}_transfer_autoencoder_US_mask_{}.json'.format(str(dataset['mask_ratios']['supervised']),
#                                                                                         str(dataset['mask_ratios']['unsupervised']))

config['results'] = {}
import os
import json
import numpy as np
import functools
import dataloader as dta
import models as mdl
import tensorflow as tf
import tensorflow_federated as tff
import datetime
from tensorboard.plugins.hparams import api as hp
    

class Algorithm(object):
    def __init__(self, opt, hps, hp_classes):
        self.opt = opt
        self.hps = hps
        self.hp_classes = hp_classes

        # self.set_log_dir('_'.join([opt['log_dir'], opt['run_number']]))
        self.run_number = self.opt['run_number']
        self.model_fp = self.opt['model_fp']
        
        self.num_rounds = self.opt['num_rounds']
        self.num_clients_per_round = self.hps[self.hp_classes['num_clients_per_round']]
        
        self.keras_model_fn = getattr(mdl, self.opt['model_fn'])(self.opt, hps, hp_classes)
        self.preprocess_fn = getattr(dta, self.opt['preprocess_fn'])
        self.dataloader = dta.DataLoader(
                                        self.preprocess_fn,
                                        self.hps[self.hp_classes['num_epochs']],
                                        self.opt['shuffle_buffer'],
                                        self.hps[self.hp_classes['batch_size']]
                                        )

    # def set_log_dir(self,directory_path):
    #     self.log_dir = directory_path
    #     if (not os.path.isdir(self.log_dir)):
    #         os.makedirs(self.log_dir)



class SupervisedLearning(Algorithm):
    def __init__(self, opt, hps, hp_classes):
        Algorithm.__init__(self, opt, hps, hp_classes)
    
    def run(self):
        run_dir = os.path.join([self.opt['log_dir'], 'run_{}'.format(str(self.opt['run_number'])]))

        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(self.hps)

            train_client_data, test_dataset = dta.get_client_data(dataset_name = self.hps[self.hp_classes['dataset']],
                                                        mask_by = self.hps[self.hp_classes['mask_by']],
                                                        mask_ratios = {'unsupervised': self.hps[self.hp_classes['unsupervised_mask_ratio']],
                                                                        'supervised': self.hps[self.hp_classes['supervised_mask_ratio']]
                                                        },
                                                        sample_client_data = self.opt['sample_client_data']
            )

            sample_batch = self.dataloader.get_sample_batch(train_client_data)
            model_fn = functools.partial(self.keras_model_fn.create_tff_model_fn, sample_batch)

            iterative_process = tff.learning.build_federated_averaging_process(model_fn)
            state = iterative_process.initialize()

            for round_num in range(self.num_rounds):
                sample_clients = np.random.choice(train_client_data.client_ids,
                                                size = min(self.num_clients_per_round,
                                                        len(train_client_data.client_ids)),
                                                replace=False
                                                )

                federated_train_data = self.dataloader.make_federated_data(train_client_data, sample_clients)

                state, metrics = iterative_process.next(state, federated_train_data)
                
                tf.summary.scalar('accuracy', metrics[1], step=1)

                print('round {:2d}, metrics={}'.format(round_num, metrics))

        model_fp = os.path.join(self.opt['log_dir'], self.opt['model_fp'].format(self.run_number))
        self.keras_model_fn.save_model_weights(model_fp, state)
        self.final_state = state
        # self.opt['results']['train_date'] = str(datetime.datetime.now())
        return




    # def solve(self, train_client_data):
    #     '''
    #     Trains model in federated setting.

    #     Arguments:
    #         train_client_data: ClientData, federated training dataset
    #     Returns:
    #         nothing, but trains model and saves trained model weights.
    #     '''
    #     sample_batch = self.dataloader.get_sample_batch(train_client_data)
    #     model_fn = functools.partial(self.keras_model_fn.create_tff_model_fn, sample_batch)

    #     iterative_process = tff.learning.build_federated_averaging_process(model_fn)
    #     state = iterative_process.initialize()

    #     for round_num in range(self.num_rounds):
    #         sample_clients = np.random.choice(train_client_data.client_ids,
    #                                         size = min(self.num_clients_per_round,
    #                                                 len(train_client_data.client_ids)),
    #                                         replace=False
    #                                         )

    #         federated_train_data = self.dataloader.make_federated_data(train_client_data, sample_clients)

    #         state, metrics = iterative_process.next(state, federated_train_data)
    #         print('round {:2d}, metrics={}'.format(round_num, metrics))

    #     model_fp = os.path.join(self.opt['log_dir'], self.opt['model_fp'].format(self.run_number))
    #     self.keras_model_fn.save_model_weights(model_fp, state)
    #     self.final_state = state
    #     # self.opt['results']['train_date'] = str(datetime.datetime.now())
    #     return

    # def evaluate(self, dataset, model_fp=None):
    #     '''
    #     Evaluates trained model in central server mode.
        
    #     Arguments:
    #         dataset: tf Dataset, contains all the test set examples as a single 
    #                 tf Dataset.
    #         model_fp: (optional) str, if model filepath is provided, it will load 
    #                 the model from file and evaluate on that. Otherwise, will 
    #                 evaluate the model at the last federated state.

    #     Returns:
    #         Nothing, but writes accuracy to file.
    #     '''
    #     processed_data = self.dataloader.preprocess_dataset(dataset)

    #     if model_fp:
    #         keras_model = self.keras_model_fn.load_model_weights(model_fp)
    #     else:
    #         keras_model = self.keras_model_fn()
    #         tff.learning.assign_weights_to_keras_model(keras_model, self.final_state.model)
        
    #     metrics = keras_model.evaluate(processed_data)
        
    #     if not keras_model.metrics:
    #         self.opt['results']['loss'] = metrics.item()
    #     else:
    #         self.opt['results']['loss'] = metrics[0].item()
    #         for i, m in enumerate(keras_model.metrics):
    #             self.opt['results'][m.name] = metrics[i+1].item()

    #     self.opt['results']['evaluation_date'] = str(datetime.datetime.now())
        
    #     if self.opt['verbose']:
    #         print('\n\n', self.opt['results'])

    #     evaluation_fp = os.path.join(self.opt['log_dir'], self.opt['evaluation_fp'])
    #     with open(evaluation_fp, 'w') as f:
    #         f.write(json.dumps(self.opt))

    #     return
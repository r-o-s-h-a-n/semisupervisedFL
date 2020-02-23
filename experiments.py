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
import plots


class Algorithm(object):
    '''
    Basic experimental loop algorithm. Must define a class that inherits from this
    and provides a `run` method which performs the experiment.
    '''

    def __init__(self, ph):
        self.ph = ph

        self.log_every = self.ph['log_every']
        self.curr_run_number = self.ph['curr_run_number']
        self.model_fp = self.ph['model_fp']
        
        self.keras_model_fn = getattr(mdl, self.ph['model_fn'])(self.ph)
        self.preprocess_fn = getattr(self.keras_model_fn, 'preprocess_{}'.format(self.ph['dataset']))

class SupervisedLearningFL(Algorithm):
    '''
    Federated supervised learning experiment loop. This includes self-supervised training.
    '''
    def __init__(self, ph):
        Algorithm.__init__(self, ph)
        self.num_rounds = self.ph['num_rounds']
        self.num_clients_per_round = self.ph['num_clients_per_round']
        self.dataloader = dta.DataLoader(
                                self.preprocess_fn,
                                self.ph['num_epochs'],
                                self.ph['shuffle_buffer'],
                                self.ph['batch_size'],
                                'federated'
                                )
    
    def run(self):
        run_dir = os.path.join(self.ph['log_dir'], 'run_{}'.format(str(self.ph['curr_run_number'])))

        # set up tensorboard summary writer scope
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(self.ph.get_hparams())
            
            # data loading
            train_client_data, test_dataset = dta.get_client_data(dataset_name = self.ph['dataset'],
                                                        mask_by = self.ph['mask_by'],
                                                        mask_ratios = {'unsupervised': self.ph['unsupervised_mask_ratio'],
                                                                        'supervised': self.ph['supervised_mask_ratio']
                                                        },
                                                        sample_client_data = self.ph['sample_client_data']
            )
            test_dataset = self.dataloader.preprocess_dataset(test_dataset)
            sample_batch = self.dataloader.get_sample_batch(train_client_data)
            model_fn = functools.partial(self.keras_model_fn.create_tff_model_fn, sample_batch)

            # federated training
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
                
                if not round_num % self.log_every:
                    print('\nround {:2d}, metrics={}'.format(round_num, metrics))
                    tf.summary.scalar('train_accuracy', metrics[0], step=round_num)
                    tf.summary.scalar('train_loss', metrics[1], step=round_num)

                    test_loss, test_accuracy = self.evaluate_central(test_dataset, state)
                    tf.summary.scalar('test_accuracy', test_accuracy, step=round_num)
                    tf.summary.scalar('test_loss', test_loss, step=round_num)

            print('\nround {:2d}, metrics={}'.format(round_num, metrics))
            tf.summary.scalar('train_accuracy', metrics[0], step=round_num)
            tf.summary.scalar('train_loss', metrics[1], step=round_num)

            test_loss, test_accuracy = self.evaluate_central(test_dataset, state)
            tf.summary.scalar('test_accuracy', test_accuracy, step=round_num)
            tf.summary.scalar('test_loss', test_loss, step=round_num)

        model_fp = os.path.join(run_dir, self.ph['model_fp'])
        self.keras_model_fn.save_model_weights(model_fp, state)
        return

    def evaluate_central(self, dataset, state):
        '''
        Evaluates a model in central server mode.
        
        Arguments:
            dataset: tf Dataset, contains all the test set examples as a single 
                    tf Dataset.
            state: tff state, the federated training state of the model. 
                    Contains model weights

        Returns:
            accuracy of model in state on dataset provided
        '''
        keras_model = self.keras_model_fn()
        shape = tf.data.DatasetSpec.from_value(dataset)._element_spec[0].shape
        keras_model.build(shape)
        
        tff.learning.assign_weights_to_keras_model(keras_model, state.model)
        metrics = keras_model.evaluate(dataset)
        return (metrics[0].item(), metrics[1].item())

    def evaluate_saved_model(self, dataset, model_fp=None):
        '''
        Evaluates trained model in central server mode.
        
        Arguments:
            dataset: tf Dataset, contains all the test set examples as a single 
                    tf Dataset.
            model_fp: str, if model filepath is provided, it will load 
                    the model from file and evaluate on that. Otherwise, will 
                    evaluate the model at the last federated state.

        Returns:
            Nothing, but writes accuracy to file.
        '''
        keras_model = self.keras_model_fn.load_model_weights(model_fp)
        return keras_model.evaluate(dataset)


class SupervisedLearningCentral(Algorithm):
    '''
    Performs the central server supervised learning experiment loop.
    '''
    def __init__(self, ph):
        Algorithm.__init__(self, ph)
        self.dataloader = dta.DataLoader(
                                self.preprocess_fn,
                                self.ph['num_epochs'],
                                self.ph['shuffle_buffer'],
                                self.ph['batch_size'],
                                'central'
                                )
        self.num_epochs = self.ph['num_epochs']

    def run(self):
        run_dir = os.path.join(self.ph['log_dir'], 'run_{}'.format(str(self.ph['curr_run_number'])))

        # set up tensorboard summary writer scope
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(self.ph.get_hparams())
            
            # data loading
            train_client_data, test_dataset = dta.get_client_data(dataset_name = self.ph['dataset'],
                                                        mask_by = self.ph['mask_by'],
                                                        mask_ratios = {'unsupervised': self.ph['unsupervised_mask_ratio'],
                                                                        'supervised': self.ph['supervised_mask_ratio']
                                                        },
                                                        sample_client_data = self.ph['sample_client_data']
            )

            train_dataset = train_client_data.create_tf_dataset_from_all_clients()
            train_dataset = self.dataloader.preprocess_dataset(train_dataset)
            test_dataset = self.dataloader.preprocess_dataset(test_dataset)

            # centralized training
            model = self.keras_model_fn()

            for epoch in range(self.num_epochs):
                model.fit(train_dataset)
                    
                if not epoch % self.log_every:
                    train_loss, train_accuracy = model.evaluate(train_dataset)
                    tf.summary.scalar('train_accuracy', train_accuracy, step=epoch)
                    tf.summary.scalar('train_loss', train_loss, step=epoch)

                    test_loss, test_accuracy = model.evaluate(test_dataset)
                    tf.summary.scalar('test_accuracy', test_accuracy, step=epoch)
                    tf.summary.scalar('test_loss', test_loss, step=epoch)
                    print('\nepoch {:2d}, train accuracy={} train loss={} test accuracy={} test loss={}'.format(
                                epoch, train_accuracy, train_loss, test_accuracy, test_loss))

            train_loss, train_accuracy = model.evaluate(train_dataset)
            tf.summary.scalar('train_accuracy', train_accuracy, step=epoch)
            tf.summary.scalar('train_loss', train_loss, step=epoch)

            test_loss, test_accuracy = model.evaluate(test_dataset)
            tf.summary.scalar('test_accuracy', test_accuracy, step=epoch)
            tf.summary.scalar('test_loss', test_loss, step=epoch)
            print('\n\n\nepoch {:2d}, train accuracy={} train loss={} test accuracy={} test loss={}'.format(
                                epoch, train_accuracy, train_loss, test_accuracy, test_loss))

        model_fp = os.path.join(run_dir, self.ph['model_fp'])
        model.save_weights(model_fp)
        return
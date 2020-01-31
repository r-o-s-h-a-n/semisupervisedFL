import os
import json
import numpy as np
import functools
import dataloader as dta
import architectures as arc
import tensorflow as tf
import tensorflow_federated as tff
import datetime


def save_model_weights(model_fp, create_model_fn, federated_state):
    '''
    Saves model weights to file from tff iteration state.

    Arguments:
        model_fp: str, filepath to save the model.
        create_model_fn: fnc, function that creates the keras model.
        federated_state: tff.python.common_libs.anonymous_tuple.AnonymousTuple, 
            the federated training state from which you wish to save the model weights.

    Returns:
        nothing, but saves model weights to disk
    '''
    model = create_model_fn()
    print(type(model))
    tff.learning.assign_weights_to_keras_model(model, federated_state.model)
    model.save_weights(model_fp)
    return


def load_model_weights(model_fp, create_model_fn):
    '''
    Loads the model weights to a new model object using create_model_fn.
    
    Arguments:
        model_fp: String, filepath to the saved model weights.
        create_model_fn: fnc, function that creates the keras model.

    Returns:
        model: tf.keras.Model, a keras model with weights initialized
    '''
    return create_model_fn().load_model(model_fp)
    

class Algorithm(object):
    def __init__(self, opt):
        self.set_experiment_dir(opt['exp_dir'])
        self.opt = opt
        self.num_rounds = self.opt['num_rounds']
        self.model_fn = self.opt['model_fn']
        self.num_clients_per_round = self.opt['num_clients_per_round']
        self.model_fp = self.opt['model_fp']

        self.dataloader = dta.DataLoader(
                                        self.opt['preprocess_fn'],
                                        self.opt['num_epochs'],
                                        self.opt['shuffle_buffer'],
                                        self.opt['max_client_examples'],
                                        self.opt['batch_size']
                                        )

    def set_experiment_dir(self,directory_path):
        self.exp_dir = directory_path
        if (not os.path.isdir(self.exp_dir)):
            os.makedirs(self.exp_dir)


class SupervisedLearning(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
    
    def solve(self, train_client_data):
        '''
        Trains model in federated setting.

        Arguments:
            train_client_data: ClientData, federated training dataset
        Returns:
            nothing, but trains model and saves trained model weights.
        '''
        sample_batch = self.dataloader.get_sample_batch(train_client_data)
        keras_model = functools.partial(self.model_fn, self.opt['model_opt'])
        model_fn = functools.partial(arc.create_tff_model_fn, keras_model, sample_batch)

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
            print('round {:2d}, metrics={}'.format(round_num, metrics))

        save_model_weights(self.model_fp, keras_model, state)
        self.final_state = state
        self.opt['results']['train_date'] = datetime.datetime.now()
        return

    def evaluate(self, dataset, model_fp=None):
        '''
        Evaluates trained model in central server mode.
        
        Arguments:
            dataset: tf Dataset, contains all the test set examples as a single 
                    tf Dataset.
            model_fp: (optional) str, if model filepath is provided, it will load 
                    the model from file and evaluate on that. Otherwise, will 
                    evaluate the model at the last federated state.

        Returns:
            Nothing, but writes accuracy to file.
        '''
        processed_data = self.dataloader.preprocess_dataset(dataset)
        print(0, type(processed_data))

        if model_fp:
            model_fn = functools.partial(self.model_fn, self.opt['model_opt'])
            keras_model = load_model_weights(model_fp, model_fn)
        else:
            keras_model = self.model_fn(self.opt['model_opt'])
            tff.learning.assign_weights_to_keras_model(keras_model, self.final_state.model)
        
        print(1, type(keras_model))
        # accuracy =  keras_model.evaluate(processed_data, as_supervised=True)
        self.opt['results']['accuracy'] = keras_model.evaluate(processed_data)
        self.opt['results']['evaluation_date'] = datetime.datetime.now()
        
        if self.opt['verbose']:
            print(self.opt['results'])

        with open(self.opt['evaluation_fp'], 'w') as f:
            f.write(json.dumps(self.opt))
        
        return
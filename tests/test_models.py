import os
import shutil
import numpy as np
import functools
import unittest
import warnings

import tensorflow as tf
import tensorflow_federated as tff

import dataloader as dta
import models as mdl

warnings.simplefilter('ignore')


class TestModels(unittest.TestCase):
    def setUp(self):
        ph = {'optimizer': 'SGD', 
                'learning_rate': 10.0,
                'dataset': 'emnist'}

        keras_model_fn = mdl.DenseAutoencoderModel(ph)
        preprocess_fn = getattr(keras_model_fn, 'preprocess_emnist')

        dataloader = dta.DataLoader(
                        preprocess_fn,
                        num_epochs=1,
                        shuffle_buffer=1,
                        batch_size=20,
                        learning_env='federated'
                        )

        train_client_data, _ = dta.get_client_data('emnist', 
                                                    'example', 
                                                    {'supervised':0.0, 
                                                    'unsupervised':0.0},
                                                    sample_client_data = True)
        
        sample_batch = dataloader.get_sample_batch(train_client_data)
        model_fn = functools.partial(keras_model_fn.create_tff_model_fn, sample_batch)

        iterative_process = tff.learning.build_federated_averaging_process(model_fn)
        state = iterative_process.initialize()

        sample_clients = train_client_data.client_ids
        federated_train_data = dataloader.make_federated_data(train_client_data, sample_clients)
        state, _ = iterative_process.next(state, federated_train_data)

        self.old_model = keras_model_fn()
        tff.learning.assign_weights_to_keras_model(self.old_model, state.model)

        self.tmp_dir = 'tests/tmp'
        os.mkdir(self.tmp_dir)
        self.model_fp = os.path.join(self.tmp_dir, 'model.h5')
        keras_model_fn.save_model_weights(self.model_fp, state)
        
        self.new_model = keras_model_fn.load_model_weights(self.model_fp)

        ph = {'optimizer': 'SGD', 
                'learning_rate': 10.0,
                'dataset': 'emnist',
                'pretrained_model_fp': self.model_fp}

        self.transfer_model = mdl.DenseSupervisedModel(ph)()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_save_model(self):
        self.assertTrue(os.path.isfile(self.model_fp))

    def test_load_model(self):
        print(self.old_model.get_weights()) 
        print(self.new_model.get_weights())
        self.assertTrue(np.all([np.testing.assert_almost_equal(self.old_model.get_weights()[i], 
                                                            self.new_model.get_weights()[i]) for i in range(len(self.old_model.get_weights()))]))
    
    def test_transfer_model(self):
        self.assertTrue(np.testing.assert_almost_equal(self.old_model.get_weights()[0], self.transfer_model.get_weights()[0]))
        self.assertFalse(np.testing.assert_almost_equal(self.old_model.get_weights()[-2], self.transfer_model.get_weights()[-2]))

if __name__ == '__main__':
    unittest.main()
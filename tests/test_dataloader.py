import unittest
import dataloader as dta
import models as mdl
import tensorflow as tf
import warnings

warnings.simplefilter('ignore')


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.train_set, _ = dta.get_client_data('emnist', 
                                            'example', 
                                            {'supervised':0.0, 
                                            'unsupervised':0.0},
                                            )

        self.train_set = dta.get_sample_client_data(self.train_set, 2, 8)

        classifier_ph = autoencoder_ph = {'dataset': 'emnist',
                                        'optimizer': 'SGD',
                                        'learning_rate': 10.0
                                        }

        self.classifier = mdl.DenseSupervisedModel(classifier_ph)
        self.autoencoder = mdl.DenseAutoencoderModel(autoencoder_ph)

        self.dataloader_classifier = dta.DataLoader(self.classifier.preprocess_emnist,
                                            num_epochs = 1,
                                            shuffle_buffer = 500,
                                            batch_size = 2,
                                            learning_env = 'federated'
                                            )

        self.dataloader_autoencoder = dta.DataLoader(self.autoencoder.preprocess_emnist,
                                            num_epochs = 1,
                                            shuffle_buffer = 500,
                                            batch_size = 2,
                                            learning_env = 'federated'
                                            )

    def test_mask_examples(self):
        def filter_supervised(x):
            return x['is_masked_supervised']

        masked_ds = dta.mask_examples(self.train_set, 0.25, 'supervised'
                    ).create_tf_dataset_from_all_clients(
                    ).filter(filter_supervised)

        num_unmasked_examples = 0
        for _ in masked_ds:
            num_unmasked_examples += 1

        self.assertEqual(num_unmasked_examples, 4)
    
    def test_mask_client(self):
        masked_ds = dta.mask_clients(self.train_set, 0.5, 'unsupervised')

        masked_ds = masked_ds.create_tf_dataset_from_all_clients().filter(lambda x: x['is_masked_unsupervised'])

        num_unmasked_examples = 0
        for _ in iter(masked_ds):
            num_unmasked_examples += 1
        
        self.assertEqual(num_unmasked_examples, 8)

    def test_make_federated_data(self):
        masked_ds = dta.mask_examples(self.train_set, 0.5, 'supervised')
        processed_ds = self.dataloader_classifier.make_federated_data(masked_ds, masked_ds.client_ids)
        self.assertEqual(len(processed_ds), 2)

    def test_preprocess_classifier(self):
        dataset = self.train_set.create_tf_dataset_for_client(self.train_set.client_ids[0])
        processed_dataset = self.classifier.preprocess_emnist(dataset, num_epochs=1, shuffle_buffer=500, batch_size=2, learning_env='federated')
        processed_batch = iter(processed_dataset).next()

        self.assertEqual(processed_batch[0].shape.as_list(), [2, 784])
        self.assertEqual(processed_batch[1].shape.as_list(), [2, 1])

    def test_preprocess_autoencoder(self):
        dataset = self.train_set.create_tf_dataset_for_client(self.train_set.client_ids[0])
        processed_dataset = self.autoencoder.preprocess_emnist(dataset, num_epochs=1, shuffle_buffer=500, batch_size=2, learning_env='federated')
        processed_batch = iter(processed_dataset).next()

        self.assertEqual(processed_batch[0].shape.as_list(), [2, 784])
        self.assertEqual(processed_batch[1].shape.as_list(), [2, 784])

if __name__ == '__main__':
    unittest.main()
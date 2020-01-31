import os
import numpy as np
import collections
import warnings
from six.moves import range
import six
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.common_libs import py_typecheck
np.random.seed(0)


def get_client_data(dataset_nm, mask_by, mask_ratios, sample_client_data=False):
  '''
  dataset_nm -- str,          name of dataset
  mask_by -- str,             indicates if we will mask by clients or examples
  mask_ratios -- dict(float), gives mask ratios for models 
                              is of format {'supervised':0.0, 'unsupervised':0.0}
  sample_dataset -- bool,     if true, will return a small ClientData dataset
                              containing 100 clients with max 100 examples each
  '''
  assert dataset_nm in ('emnist')
  assert mask_by in ('client', 'example'), 'mask_by must be `client` or `example`'

  if dataset_nm == 'emnist':
    train_set, test_set = tff.simulation.datasets.emnist.load_data()

  if sample_client_data:
    train_set = get_sample_client_data(train_set, 100, 100)
    test_set = get_sample_client_data(test_set, 100, 100)

  for s in mask_ratios:
    if mask_by == 'examples':
      train_set = mask_examples(train_set, mask_ratios[s], s)
    else:
      train_set = mask_clients(train_set, mask_ratios[s], s)

  # flatten test set as a single tf dataset
  test_set = test_set.create_tf_dataset_from_all_clients()

  return train_set, test_set


def get_sample_client_data(client_data, num_clients, num_examples):
    def get_dataset(client_id):
      return client_data.create_tf_dataset_for_client(client_id).take(num_examples)

    return tff.simulation.client_data.ConcreteClientData(client_data.client_ids[:num_clients], get_dataset)


def mask_true(example, mask_type):
    key = 'is_masked_'+mask_type
    example[key] = tf.convert_to_tensor(True)
    return example


def mask_false(example, mask_type):
    key = 'is_masked_'+mask_type
    example[key] = tf.convert_to_tensor(False)
    return example


def mask_examples(client_data, mask_ratio, mask_type, seed=None):
    '''
    Masks mask_ratio fraction of examples uniformly randomly across all clients.

    Args:
        client_data - ClientData object containing federated dataset
        mask_ratio - float, fraction of total examples to be masked
    Returns:
        client_data - ClientData object, identical to client_data argument but 
        with additional attribute `is_masked` boolean for each example
    '''
    assert mask_type in ('supervised', 'selfsupervised'), 'mask type must be `supervised` or `selfsupervised`'

    def get_example_ids_generator():
      # generates unique tuple for each example of structure (client_id, example_id)
      for client_id in client_data.client_ids:
        for i, _ in enumerate(client_data.create_tf_dataset_for_client(client_id)):
            yield (client_id, i)
    
    # generate example ids, shuffle, then select the ones to be masked
    example_ids = [x for x in get_example_ids_generator()]
    # np.random.RandomState(seed=seed).shuffle(example_ids)
    np.random.shuffle(example_ids)
    num_examples = len(example_ids)
    masked_example_idxs = example_ids[:int(mask_ratio*num_examples)]

    # convert example ids to a dict mapping client ids to a tensor of example ids chosen to be masked
    masked_example_idxs_dict = collections.defaultdict(list)
    for ex in masked_example_idxs:
        client_id, example_id = ex
        masked_example_idxs_dict[client_id].append(example_id)
    
    for client_id in masked_example_idxs_dict:
        masked_example_idxs_dict[client_id] = tf.convert_to_tensor(masked_example_idxs_dict[client_id], dtype=tf.int64)

    def preprocess_fn(dataset, client_id):
        return dataset.enumerate().map(lambda i, x: mask_true(x, mask_type)
                                       if tf.reduce_any(tf.math.equal(i, masked_example_idxs_dict[client_id]))
                                       else mask_false(x, mask_type))
        
    tff.python.common_libs.py_typecheck.check_callable(preprocess_fn)

    def get_dataset(client_id):
      return preprocess_fn(client_data.create_tf_dataset_for_client(client_id), client_id)

    return tff.simulation.client_data.ConcreteClientData(client_data.client_ids, get_dataset)


def mask_clients(client_data, mask_ratio, mask_type, seed=None):
    '''
    Masks mask_ratio fraction of clients uniformly randomly. If a client is 
    selected as masked, all examples it contains are treated as masked.
    
    Args:
        client_data - ClientData object containing federated dataset
        mask_ratio - float, fraction of total clients to be masked
    Returns:
        client_data - ClientData object, identical to client_data argument but 
        with additional attribute `is_masked` boolean for each example
    '''
    # get client idxs to mask
    client_ids = list(client_data.client_ids)
    np.random.shuffle(client_ids)
    num_clients = len(client_data.client_ids)
    masked_client_idxs = set(client_ids[:int(mask_ratio*num_clients)])

    def preprocess_fn(dataset, client_id):
        if client_id in masked_client_idxs:
          return dataset.map(lambda x: mask_true(x, mask_type))
        else:
          return dataset.map(lambda x: mask_false(x, mask_type))

    tff.python.common_libs.py_typecheck.check_callable(preprocess_fn)

    def get_dataset(client_id):
      return preprocess_fn(client_data.create_tf_dataset_for_client(client_id), client_id)

    return tff.simulation.client_data.ConcreteClientData(client_data.client_ids, get_dataset)


def preprocess_classifier(dataset, 
                        num_epochs, 
                        shuffle_buffer, 
                        max_examples_per_client, 
                        batch_size):

  def element_fn(element):
    return collections.OrderedDict([
          ('x', tf.reshape(element['pixels'], [-1])),
          ('y', tf.reshape(element['label'], [1]))
      ])

  # filter by `is masked` if `is masked` is an attribute of the element
  return dataset.filter(lambda x: not x['is_masked_supervised'] if 'is_masked_supervised' in x else True).repeat(
      num_epochs).map(element_fn).shuffle(shuffle_buffer).take(
          max_examples_per_client).batch(batch_size)


def preprocess_autoencoder(dataset,
                        num_epochs, 
                        shuffle_buffer, 
                        max_examples_per_client, 
                        batch_size):

  def element_fn(element):
    return collections.OrderedDict([
          ('x', tf.reshape(element['pixels'], [-1])),
          ('y', tf.reshape(element['pixels'], [-1])),
      ])

  return dataset.filter(lambda x: not x['is_masked_selfsupervised'] if 'is_masked_selfsupervised' in x else True).repeat(
      num_epochs).map(element_fn).shuffle(shuffle_buffer).take(
          max_examples_per_client).batch(batch_size)


class DataLoader(object):
    def __init__(self, 
                preprocess_fn,
                num_epochs = 10, 
                shuffle_buffer = 500, 
                max_examples_per_client = 10000, 
                batch_size = 128
                ):
        self.preprocess_fn = preprocess_fn
        self.num_epochs = num_epochs
        self.shuffle_buffer = shuffle_buffer
        self.max_examples_per_client = max_examples_per_client
        self.batch_size = batch_size
      
    def preprocess_dataset(self, dataset):
        return self.preprocess_fn(dataset,
                                  self.num_epochs,
                                  self.shuffle_buffer,
                                  self.max_examples_per_client,
                                  self.batch_size
                                  )

    def make_federated_data(self, client_data, client_ids):
        return [self.preprocess_dataset(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

    def get_sample_batch(self, client_data):
        preprocessed_example_dataset = self.preprocess_dataset(
                  client_data.create_tf_dataset_for_client(client_data.client_ids[0]))

        return tf.nest.map_structure(
            lambda x: x.numpy(), iter(preprocessed_example_dataset).next())
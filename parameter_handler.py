from pprint import pprint
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


class ParameterHandler(object):
    '''
    Stores and provides experiment and model settings including global configs, hyperparameters
    and experimental parameters.
    '''
    def __init__(self, config, hparam_map, metrics_map):
        # config is a dictionary containing experiment and model settings that will not vary across experiments
        self.config = config
        # hparam_map is is a dictionary from hyperparamter name, as str, to tensorboard HParam instances
        # which describe the range of possible values of the hyperparameter
        self.hparam_map = hparam_map
        # metrics_map is a dictionary from metric name, as str, to tensorboard Metrics instances
        # which describe the metric to keep track of later
        self.metrics_map = metrics_map
        # hparams are the specific values of the hyperparameters being used in the current instance of the model
        # this gets set in the main.py script
        self.hparams = None

    def _init_hparams(self):
        with tf.summary.create_file_writer(self.config['log_dir']).as_default():
            hp.hparams_config(
                hparams=self.hparam_map.values(),
                metrics=self.metrics_map.values(),
            )

    def __setitem__(self, key, value):
        '''
        A way to quickly add or change value of items to the model config after instantiating the model.
        Must go in config, since you cannot modify hyperparamters after starting the experiment.
        '''
        self.config[key] = value

    def set_hparams(self, hparams, curr_run_number=0):
        '''
        Sets the current set of chosen hyperaparamters to run the experiment on.

        Arugments:
            hparams: dictionary of hyperparameters to current setting
            curr_run_number: int, the index of the hyperparameter combination set that this experiment is running.
                    For example, if there are 2 hyperparameters, each with 2 possible values, there would be 
                    a total of 4 experiments and curr_run_number would vary across the range [0, 4). If hparams
                    are generated using the `gen_hparam_cartesian_product` method, the order of hparams
                    generated will be deterministic and one can stop the experiment and start at the last
                    curr_run_number by setting the curr_run_number in the config file.
        
        Returns:
            nothing, but sets the hyperparameters and curr_run_number
        '''
        print('\n\n--- RUN NUMBER {} ---\n'.format(curr_run_number))
        pprint(hparams)
        print('\n')

        self.hparams = hparams
        self.config['curr_run_number'] = curr_run_number
    
    def __getitem__(self, key):
        '''
        Finds the value for a given parameter. It first checks to see if the parameter is a hyperparamter.
        It then checks if it is in config, as a global model paramter. Finally it raises KeyError if it 
        cannot find a parameter with that name.

        Arguments:
            key: str, the parameter name
        
        Returns:
            value of the parameter in the current experiment.
        '''
        # search hp first
        if key in self.hparam_map:
            return self.hparams[self.hparam_map[key]]
        elif key in self.config:
            return self.config[key]
        else:
            raise KeyError('No saved parameter named `{}`'.format(key))

    def get_hparams(self):
        if not self.hparams:
            raise AttributeError('need to generate a set of hyperpamarameters for this experiment first')
        return self.hparams
    
    def get_metrics(self):
        return self.metrics_map

    def gen_hparam_cartesian_product(self):
        '''
        Generates the cartesian product of possible hyperparamter values.
        Arguments:
            None, but takes self.hparam_map
        Returns:
            Generator, yields the set of hyperparamter set for an individual experiment
                The order of the experiments is deterministic.
        '''
        hps = sorted(list(self.hparam_map.values()), key=lambda x: str(x))
        def helper(i, sofar={}):
            for val in hps[i].domain.values:
                next = {k:sofar[k] for k in sofar}
                next[hps[i]] = val
                if i == len(hps)-1:
                    yield next
                else:
                    for x in helper(i+1, next):
                        yield x
        return helper(0)
    
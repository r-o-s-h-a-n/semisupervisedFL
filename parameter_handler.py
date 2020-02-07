from pprint import pprint
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


class ParameterHandler(object):
    def __init__(self, config, hparam_map, metrics_map):
        self.config = config
        self.hparam_map = hparam_map
        self.metrics_map = metrics_map
        self.hparams = None

    def _init_hparams(self):
        with tf.summary.create_file_writer(self.config['log_dir']).as_default():
            hp.hparams_config(
                hparams=self.hparam_map.values(),
                metrics=self.metrics_map.values(),
            )

    def __setitem__(self, key, value):
        self.config[key] = value

    def set_hparams(self, hparams, run_number=0):
        print('\n\n--- RUN NUMBER {} ---\n'.format(run_number))
        pprint(hparams)
        print('\n')

        self.hparams = hparams
        self.config['run_number'] = run_number
    
    def __getitem__(self, key):
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
    
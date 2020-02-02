# semisupervisedFL
Semisupervised and transfer learning trained in federated setting.

To set up your environment, run the following:
1. `pip install virtualenv`
1. `virtualenv env` (or `virtualenv <name of your environment>`
2. `source setup.sh` 
and you're good to go!

To run an existing experiment, simply run `python main.py --exp <experiment_name>`. For example to run a simple classifier on the EMNIST dataset, run `python main.py --exp supervised`. The attribute `experiment_name` must correspond to a config file in the folder `configs`. 

To build a new experiment you may wish to make the following modifications:
1. Add a new config file for a new experiment specifying different model and data configurations.
2. Add a new model to the file `models.py`. A new model must inherit from the class `Model` and specify a `__call__` method. The `__call__` method must return a compiled tf keras model.
3. Add a new experiment loop algorithm to `experiments.py`. Currently only a supervised training experiment loop is specified. It is likely that this algorithm can be extended to self supervised and semisupervised settings by just swapping the model in `models.py`. In addition, if you are using a new model that has different data inputs and targets, you will need to specify a new preprocess function in `dataloader.py`. 

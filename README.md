# semisupervisedFL
Semisupervised and transfer learning trained in federated setting.

Experiments were run using:
* Python 3.7.4
* tensorflow 2.1.0
* tensorflow-federated 0.12.0

To set up your environment, run the following:
1. `pip install virtualenv`
2. `virtualenv env` (or `virtualenv <name of your environment>`)
3. `pip install -r requirements.txt` 

and you're good to go!

### Run an existing experiment
To run an existing experiment, run `python main.py --exp <experiment_name>`. For example to run a federated classifier on the EMNIST dataset, run `python main.py --exp dense_emnist_federated_supervised`. The attribute `experiment_name` must correspond to a config file in the folder `configs`. 

Alternatively if you would like to run an experiment over a set of hyperparameters, run `source run.sh <experiment_name>` to run the experiments over the cartesian product of hyperparameters provided in the config file.

### Monitor results in Tensorboard
To view the results in Tensorboard, run `tensorboard --logdir=logs/<experiment_name> --port=<port number>`. If running locally, follow the instructions in terminal to connect to `http://localhost:<port number>`. If running remotely, set up port forwarding by running the following on your local machine: `ssh -NfL <port number>:localhost:<port number> -i <path to pem file> <username>@<remote IP address>`. Then go to `http://localhost:<port number>` locally.

### Build a new experiment
To build a new experiment you may wish to make the following modifications:
1. Add a new config file to `config/` for a new experiment specifying different hyperparameters you want to try as well as model and data configurations.
2. Add a new model to the file `models.py`. A new model must inherit from the class `Model` and specify a `__call__` method and a `preprocess` method. The `__call__` method must return a compiled tf keras model. The `preprocess` method must preprocess a tf Dataset to be trained by the model.
3. Add a new experiment loop algorithm to `experiments.py`. Currently only a supervised training experiment loop is specified. It is likely that this algorithm can be extended to self supervised and semisupervised settings by just swapping the model in `models.py`.

### Outline of Repository ###
`models/` contains models you want to try running an experiment on
`config/` contains config files. Config files contain experimental parameters for the specific experiment you want to run.
`tests/` contains tests for some of the operations in the repo
`dataloader.py` contains functions for loading data into models
`experiments.py` contains experiment loops
`main.py` is the script that executes an experiment by applying a config file to an experiment loop

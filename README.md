# semisupervisedFL
Semisupervised and transfer learning trained in federated setting.

To set up your environment, run the following:
1. `pip install virtualenv`
2. `virtualenv env` (or `virtualenv <name of your environment>`)
3. `source setup.sh` 

and you're good to go!

### Run an existing experiment
To run an existing experiment, simply run `python main.py --exp <experiment_name>`. For example to run a simple classifier on the EMNIST dataset, run `python main.py --exp supervised`. The attribute `experiment_name` must correspond to a config file in the folder `configs`. 

### Monitor results in Tensorboard
To view the results in Tensorboard, run `tensorboard --logdir=logs/<experiment_name> --port=<port number>`. If running locally, follow the instructions in terminal to connect to `http:localhost:<port number>`. If running remotely, set up port forwarding by running the following on your local machine: `ssh -NfL <port number>:localhost:<port number> -i <path to pem file> <username>@<remote IP address>`. Then go to `http:localhost:<port number>` locally.

Note: At the time of writing this README, Tensorboard only runs on the latest stable release of Tensorflow, and breaks on the nightly release of Tensorflow. To run Tensorboard:
1. Deactivate any currently running virtual environments by running `deactivate`
2. Create a new virtual environment for Tensorboard by running `virtualenv tboard`
3. Run `source setup_tboard.sh`

### Build a new experiment
To build a new experiment you may wish to make the following modifications:
1. Add a new config file to `config/` for a new experiment specifying different hyperparameters you want to try as well as model and data configurations.
2. Add a new model to the file `models.py`. A new model must inherit from the class `Model` and specify a `__call__` method. The `__call__` method must return a compiled tf keras model.
3. Add a new experiment loop algorithm to `experiments.py`. Currently only a supervised training experiment loop is specified. It is likely that this algorithm can be extended to self supervised and semisupervised settings by just swapping the model in `models.py`. In addition, if you are using a new model that has different data inputs and targets, you will need to specify a new preprocess function in `dataloader.py`. 

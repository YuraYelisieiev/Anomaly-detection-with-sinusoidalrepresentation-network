# Basic guide to repository

## Experiment configuration and start
All experiment settings are done in the config file ``config.yaml``. 
If you want to add multi GPU training, dp or ddp and the bunch of other training tricks you can add parameters to the Trainer in ``train.py``

Here is a list of all parameters [link](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

You can also choose to train experiment with or without discriminator.

After the setup you can simply run an experiment ```python3 train.py```.

## Modules 
All training logic is placed in the modules folder. PyTorch Lightning will do everything yourself,
the main thing is simple, but if you need to use optimization by yourself you can read this 
[article](https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html?highlight=Manual%20optimization#manual-optimization).

 - Network architectures - ``models.py``
 - Data operations - ``data.py``
 - Image formation for tensorboard - ``utils.py``

## Data
You can feed any dataset into the model, but first you need to split the data into 2 folders and 
change config parameters of `train` and `val` dataset paths.

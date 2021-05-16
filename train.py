from pytorch_lightning import Trainer, seed_everything
from data import RIVETSDataModule
from modules import RGAN, Reconstruction
from callbacks import CheckpointEveryNSteps
import yaml

if __name__ == "__main__":
    # seed_everything(42)
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = RGAN(config)
    datamodule = RIVETSDataModule(config)
    trainer = Trainer(deterministic=True,
                      gpus=[0],
                      max_epochs=100000,
                      callbacks=[CheckpointEveryNSteps()])
    trainer.fit(model, datamodule)


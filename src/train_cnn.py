"""Script for model training"""

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
#from config.constants import NUM_WORKERS
NUM_WORKERS=0

if __name__ == "__main__":

    from data.dataset import Dataset
    from modules.lightning_cnn import LitCNN

    dataset = Dataset()
    lightning_cnn = LitCNN()
    name = "brain_model"

    train_dataset, val_dataset, test_dataset = dataset.get_dataset()

    train_loader = DataLoader(
        train_dataset,
        num_workers=NUM_WORKERS,
        batch_size=16,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=NUM_WORKERS,
    )

    trainer = Trainer(accelerator="gpu", devices=1,
        logger=TensorBoardLogger(
            "/home/ubuntu/giorgio/v311/lightning_logs",
            name=name,
            log_graph=True,
        ),  # pyright: ignore[reportArgumentType]
        callbacks=[  # pyright: ignore[reportGeneralTypeIssues]
            EarlyStopping(monitor="val_mse_loss", mode="min", patience=10),
            ModelSummary(
                max_depth=-1
            ),  # print the weights summary of the model when trainer.fit() is called
            LearningRateMonitor(logging_interval="epoch"),
        ],
        max_epochs=300,
        log_every_n_steps=1,
    )

    trainer.fit(
        model=lightning_cnn, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    trainer.test(model=lightning_cnn, dataloaders=test_loader)
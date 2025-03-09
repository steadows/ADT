# train_rna2adt.py

# Adds phase 4 cross modal training.
# My phase 4 monitoring metric is cross modal total loss.
# Changed learning rate to .001
# Changed mps to cpu but can change to gpu or mps if u want.


import os
import shutil
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from data_module import RNAADTDataModule
from model import CrossModalVAE
from callbacks import SharedEmbeddingCallback


if __name__ == "__main__":
    pl.seed_everything(1986)

    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    import multiprocessing

    multiprocessing.freeze_support()  # For Windows/macOS multiprocessing

    # Optionally, clear previous logs/checkpoints if you want a fresh start:
    log_dir = "checkpoints"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    # Set precision if needed
    torch.set_float32_matmul_precision("medium")

    def get_callbacks(phase):
        # Configure monitor metric and mode based on phase:
        if phase == "rna":
            monitor_metric = "rna_validation_reconstruction_loss"
            mode = "min"
        elif phase == "adt":
            monitor_metric = "adt_validation_reconstruction_loss"
            mode = "min"
        elif phase == "rna_to_adt":
            monitor_metric = "rna_to_adt_validation_pearson"
            mode = "max"
        elif phase == "adt_to_rna":
            monitor_metric = "adt_to_rna_validation_pearson"
            mode = "max"
        else:
            raise ValueError("Invalid phase")

        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"best_model_{phase}",
            monitor=monitor_metric,
            save_top_k=1,
            mode=mode,
            save_last=True,
        )
        early_stop_callback = EarlyStopping(
            monitor=monitor_metric,
            min_delta=0.001,
            patience=10,
            verbose=True,
            mode=mode,
        )
        return [checkpoint_callback, early_stop_callback]

    # Hyperparameters
    lr = 0.001  # Adjust as needed
    batch_size = 120  # You can experiment with batch size
    num_workers = 4  # Adjust based on your system
    data_dir = "data/"  # Make sure your CSV files are here

    # Instantiate the model
    model = CrossModalVAE(lr=lr)

    # Instantiate DataModule with batch size and number of workers
    dm = RNAADTDataModule(
        batch_size=batch_size, num_workers=num_workers, data_dir=data_dir
    )
    dm.setup()

    # Initialize logger (TensorBoardLogger is used if tensorboard is installed, otherwise CSVLogger)
    logger = TensorBoardLogger(
        "TensorBoard", name="logs", log_graph=True, default_hp_metric=False
    )

    # ---------------------------
    # PHASE 1: RNA Autoencoder Training
    print("\n--- Phase 1: RNA Autoencoder Training ---\n")
    model.phase = "rna"
    dm.mode = "rna"
    callbacks = get_callbacks("rna")
    trainer = pl.Trainer(
        max_epochs=150,
        accelerator=accelerator,  # Change to "gpu" or "mps" if available
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
    )
    trainer.fit(model, dm)

    # ---------------------------
    # PHASE 2: ADT Autoencoder Training
    print("\n--- Phase 2: ADT Autoencoder Training ---\n")
    model.phase = "adt"
    dm.mode = "adt"
    callbacks = get_callbacks("adt")
    trainer = pl.Trainer(
        max_epochs=150,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
    )
    trainer.fit(model, dm)

    # ---------------------------
    # PHASE 3: Cross-Modal Training (RNA -> ADT)
    print("\n--- Phase 3: Cross-Modal Training (RNA -> ADT) ---\n")
    model.phase = "rna_to_adt"
    dm.mode = "cross_modal"
    callbacks = get_callbacks("rna_to_adt")
    embedding_callback = SharedEmbeddingCallback()

    trainer = pl.Trainer(
        max_epochs=150,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks + [embedding_callback],
        logger=logger,
        log_every_n_steps=5,
    )
    trainer.fit(model, dm)

    # ---------------------------
    # PHASE 4: Cross-Modal Training (ADT -> RNA)
    print("\n--- Phase 4: Cross-Modal Training (ADT -> RNA) ---\n")
    model.phase = "adt_to_rna"
    dm.mode = "cross_modal"
    callbacks = get_callbacks("adt_to_rna")
    trainer = pl.Trainer(
        max_epochs=150,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks + [embedding_callback],
        logger=logger,
        log_every_n_steps=5,
    )
    trainer.fit(model, dm)

    # ---------------------------
    # PHASE 5: Cross-Modal Evaluation
    print("\n--- Phase 5: Cross-Modal Evaluation ---\n")

    # 1. Instantiate your LightningModule from the best checkpoint
    model = CrossModalVAE.load_from_checkpoint(
        "checkpoints/best_model_rna_to_adt.ckpt"
    )

    trainer = pl.Trainer(
        accelerator=accelerator, 
        devices=1, 
        logger=logger, 
        log_every_n_steps=5
    )
    trainer.test(model, dm)

    print("\nTraining Complete! Check your logs and checkpoints for details.")

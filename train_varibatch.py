# train_rna2adt.py

# Adds phase 4 cross modal training.
# My phase 4 monitoring metric is cross modal total loss.
# Changed learning rate to .001
# Changed mps to cpu but can change to gpu or mps if u want.


import os
import shutil
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from data_module_varibatch import RNAADTDataModule
from model_varibatch import CrossModalVAE
from callbacks import SharedEmbeddingCallback
from pytorch_lightning.strategies import DeepSpeedStrategy  # Using DeepSpeed
from pytorch_lightning.utilities import rank_zero_only
import logging
# Only log warnings and above for Lightning on all ranks
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


@rank_zero_only
def my_print(message):
    print(message)

# Now only rank 0 will execute this print:
my_print("This message is printed only by rank 0.")



if __name__ == "__main__":
    start_time = time.time()
    pl.seed_everything(1986)

    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    @rank_zero_only
    def my_print(message):
        print(message)
    
    # Now only rank 0 will execute this print:
    my_print("This message is printed only by rank 0.")    

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
        if phase == "autoencoder":
            monitor_metric = "combined_val_loss"  # Track both RNA & ADT
            mode = "min"
        elif phase == "cross_modal":
            monitor_metric = "crossmodal_rna2adt_val_pearson"  # Only track RNA → ADT
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
    batch_size = 632  # You can experiment with batch size
    kl_weight_rna = 1e-6  # RNA autoencoder
    kl_weight_adt = 1e-6  # ADT autoencoder
    kl_weight_cross = 1e-5  # Cross-modal (a little stronger)
    num_workers = 4  # Adjust based on your system
    data_dir = "data/"  # Make sure your CSV files are here
    
    # Define KL annealing schedule here ("linear", "sigmoid", "exponential", "none")
    kl_schedule = "exponential"

    # Instantiate the model
    model = CrossModalVAE(lr=lr)
    model.kl_schedule = kl_schedule
    model.kl_weight_rna = kl_weight_rna
    model.kl_weight_adt = kl_weight_adt
    model.kl_weight_cross = kl_weight_cross

    # Instantiate DataModule with batch size and number of workers
    dm = RNAADTDataModule(
        batch_size=batch_size, num_workers=num_workers, data_dir=data_dir
    )
    dm.setup()

    # Initialize logger (TensorBoardLogger is used if tensorboard is installed, otherwise CSVLogger)
    logger = TensorBoardLogger(
        "TensorBoard", name="logs", log_graph=True, default_hp_metric=False
    )

    # Define your DeepSpeed configuration explicitly:
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 632  # Set this to your per-GPU batch size
        # You can include additional DeepSpeed options here if needed
    }
        
    # ---------------------------
    # PHASE 1: Autoencoder Training
    print("\n--- Phase 1: Autoencoder Training ---\n")
    model.phase = "autoencoder"
    dm.mode = "combined"
    callbacks = get_callbacks("autoencoder")
    trainer = pl.Trainer(
        max_epochs=150,
        accelerator=accelerator,  # Change to "gpu" or "mps" if available
        devices=8,
        accumulate_grad_batches=1,
        strategy=DeepSpeedStrategy(stage=2, config=deepspeed_config), 
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
    )
    trainer.fit(model, dm)

    # Define your DeepSpeed configuration explicitly:
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 632  # Set this to your per-GPU batch size
        # You can include additional DeepSpeed options here if needed
    }
    
    # PHASE 2: Cross-Modal Training
    print("\n--- Phase 2: Cross-Modal Training ---\n")
    model.phase = "cross_modal"
    dm.mode = "combined"
    callbacks = get_callbacks("cross_modal")
    embedding_callback = SharedEmbeddingCallback()

    trainer = pl.Trainer(
        max_epochs=150,
        accelerator=accelerator,
        devices=8,
        accumulate_grad_batches=1,
        strategy=DeepSpeedStrategy(stage=2, config=deepspeed_config),
        callbacks=callbacks + [embedding_callback],
        logger=logger,
        log_every_n_steps=5,
    )
    trainer.fit(model, dm)

    # Define your DeepSpeed configuration explicitly:
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 632  # Set this to your per-GPU batch size
        # You can include additional DeepSpeed options here if needed
    }

    # ---------------------------
    # PHASE 3: Cross-Modal Evaluation
    print("\n--- Phase 3: Cross-Modal Evaluation ---\n")

    # 1. Instantiate your LightningModule from the best checkpoint
    model = CrossModalVAE.load_from_checkpoint(
        "/home/ubuntu/ADT/checkpoints/best_model_cross_modal.ckpt",
        lr=lr,
    )

    trainer = pl.Trainer(
        accelerator=accelerator, 
        devices=8,
        accumulate_grad_batches=1,
        strategy=DeepSpeedStrategy(stage=2, config=deepspeed_config), 
        logger=logger, 
        log_every_n_steps=5
    )
    trainer.test(model, dm)

    end_time = time.time()  # End time for runtime measurement
    total_runtime = end_time - start_time
    print(f"\nTraining Complete! Total runtime: {total_runtime:.2f} seconds.")

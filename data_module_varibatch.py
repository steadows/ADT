# data_module.py
# Key changes: Adds a cross_modal mode, allowing the model to learn RNA -> ADT and ADT-> RNA mappings
# When cross modal is selected, we use the unpaired cross modal dataset.


# Can we use the validation data loader for cross modal and somehow consolidate bc were basically writing the same thign twice?

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import torch
from dataset import RNAADT_Dataset,PairedEvalDataset
from sklearn.model_selection import train_test_split
from lightning.pytorch.utilities.combined_loader import CombinedLoader



class RNAADTDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size=128, num_workers=4, data_dir="data/", mode="autoencoder"
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.mode = mode  # "rna", "adt", "autoencoder", or "cross_modal"

    def setup(self, stage=None):
        # Load CSV files
        train_rna = pd.read_csv(f"{self.data_dir}/train_rna.csv")
        train_adt = pd.read_csv(f"{self.data_dir}/train_adt.csv")

        eval_rna = pd.read_csv(f"{self.data_dir}/eval_rna.csv" )
        eval_adt = pd.read_csv(f"{self.data_dir}/eval_adt.csv")

        test_rna = pd.read_csv(f"{self.data_dir}/test_rna.csv")
    
        

        # Transpose if needed
        train_rna = train_rna.T
        train_adt = train_adt.T
        eval_rna = eval_rna.T
        eval_adt = eval_adt.T
        test_rna = test_rna.T
        
        # Store sample names (or feature names, if that’s what you mean)
        self.rna_sample_names = train_rna.index.tolist()
        self.adt_sample_names = train_adt.index.tolist()

        # Convert to tensors
        self.train_rna = torch.tensor(train_rna.values, dtype=torch.float32)
        self.train_adt = torch.tensor(train_adt.values, dtype=torch.float32)
        self.eval_rna = torch.tensor(eval_rna.values, dtype=torch.float32)
        self.eval_adt = torch.tensor(eval_adt.values, dtype=torch.float32)
        self.test_rna = torch.tensor(test_rna.values, dtype=torch.float32)

        # Create 80/20 train/validation split
        num_rna_samples = self.train_rna.shape[0]
        num_adt_samples = self.train_adt.shape[0]
        rna_indices = torch.arange(num_rna_samples)
        adt_indices = torch.arange(num_adt_samples)
        
        rna_train_idx, rna_val_idx = train_test_split(
            rna_indices, test_size=0.2, shuffle=True, random_state=1986
        )
        adt_train_idx, adt_val_idx = train_test_split(
            adt_indices, test_size=0.2, shuffle=True, random_state=1986
        )
        self.train_rna_train = self.train_rna[rna_train_idx]
        self.train_rna_val = self.train_rna[rna_val_idx]
        self.train_adt_train = self.train_adt[adt_train_idx]
        self.train_adt_val = self.train_adt[adt_val_idx]
        
         # Also, split the sample names so that they match the training and validation data
        self.train_rna_sample_names = [self.rna_sample_names[i] for i in rna_train_idx]
        self.val_rna_sample_names = [self.rna_sample_names[i] for i in rna_val_idx]
        self.train_adt_sample_names = [self.adt_sample_names[i] for i in adt_train_idx]
        self.val_adt_sample_names = [self.adt_sample_names[i] for i in adt_val_idx]

        # print("train_rna_train shape:", self.train_rna_train.shape)
        # print("train_rna_val shape:  ", self.train_rna_val.shape)
        # print("train_adt_train shape:", self.train_adt_train.shape)
        # print("train_adt_val shape:  ", self.train_adt_val.shape)

    def train_dataloader(self):
        if self.mode == "rna":
            return DataLoader(
                self.train_rna_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=False,
                persistent_workers=True,
            )
        elif self.mode == "adt":
            return DataLoader(
                self.train_adt_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=False,
                persistent_workers=True,
            )
        elif self.mode == "combined":  # New mode using CombinedLoader
            rna_loader = DataLoader(
                self.train_rna_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=False,
                persistent_workers=True,
            )
            adt_loader = DataLoader(
                self.train_adt_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=False,
                persistent_workers=True,
            )
            iterables = {'rna': rna_loader, 'adt': adt_loader}
            # 'max_size_cycle' will cycle the shorter loader to match the longest one
            return CombinedLoader(iterables, mode='max_size_cycle')
        else:
            raise ValueError("Invalid mode for DataModule")

    def val_dataloader(self):
        if self.mode == "rna":
            return DataLoader(
                self.train_rna_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                persistent_workers=True,
            )
        elif self.mode == "adt":
            return DataLoader(
                self.train_adt_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                persistent_workers=True,
            )
        elif self.mode == "combined":
            rna_loader = DataLoader(
                self.train_rna_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                persistent_workers=True,
            )
            adt_loader = DataLoader(
                self.train_adt_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                persistent_workers=True,
            )
            iterables = {'rna': rna_loader, 'adt': adt_loader}
            return CombinedLoader(iterables, mode='min_size')
        else:
            raise ValueError("Invalid mode for DataModule")
                
    def test_dataloader(self):
        # This DataLoader uses the paired evaluation data (for sanity checking)
        dataset = PairedEvalDataset(self.eval_rna, self.eval_adt)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )

    def submission_dataloader(self):
        # This DataLoader uses only the test RNA data (for final submission predictions)
        return DataLoader(
            self.test_rna,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )

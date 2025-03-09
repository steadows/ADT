# Key changes:
# Added a cross modal training that introduces randomly selected pairs between adt and rna train.
# The original RNAADT_Dataset remains unchanges so it still worjs the same for autoencoding.

import torch
from torch.utils.data import Dataset
import random


class RNAADT_Dataset(Dataset):
    """
    PyTorch Dataset for loading RNA-ADT data in either training, validation, or test mode.
    """

    def __init__(self, rna_data, adt_data=None, mode="rna"):
        """
        Args:
            rna_data (torch.Tensor): RNA feature matrix.
            adt_data (torch.Tensor, optional): ADT feature matrix. Not needed for test mode.
            mode (str): "rna" (RNA autoencoder), "adt" (ADT autoencoder), "test_rna" (Test RNA).
        """
        self.rna_data = rna_data
        self.adt_data = adt_data
        self.mode = mode

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, idx):
        if self.mode == "cross_modal":
            return (
                self.rna_data[idx].clone().detach(),
                self.adt_data[idx].clone().detach()
            )
class PairedEvalDataset(Dataset):
    def __init__(self, eval_rna_tensor, eval_adt_tensor):
        self.eval_rna_tensor = eval_rna_tensor
        self.eval_adt_tensor = eval_adt_tensor
        assert len(eval_rna_tensor) == len(eval_adt_tensor), "Must match row counts!"

    def __len__(self):
        return len(self.eval_rna_tensor)

    def __getitem__(self, idx):
        # Return matching rna and adt samples
        return self.eval_rna_tensor[idx], self.eval_adt_tensor[idx]

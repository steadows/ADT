# model.py
# Key changes: Added cross modal phase explicitly in the training step.
# Explicitly computing loss for rna to adt and adt to rna.
# Added KL divergence to both RNA and ADT in cross modal training.
# Retained batch normalization in the final adt decoder layer
# Kept relu in adt decoder last activation

# Investigate during the validation step whether we can use the eval dataset for validation (This will be in the data module and model .py (specifically the eval dataloader)).

import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F


def pearson_corrcoef(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
    )
    return corr


def r2_score(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - ss_res / ss_tot


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class CrossModalVAE(pl.LightningModule):
    def __init__(self, lr, rna_dim=10000, adt_dim=25):
        super().__init__()
        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self.latent_dim = 64
        self.phase = "autoencoder"  # Default phase

        # RNA Encoder
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim * 2),
        )

        # ADT Encoder
        self.adt_encoder = nn.Sequential(
            nn.Linear(adt_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim * 2),
        )

        # Shared latent transformation
        self.shared_transform = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
        )

        # RNA Decoder
        self.rna_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, rna_dim),
            nn.BatchNorm1d(rna_dim),
            nn.ReLU(),
        )

        # ADT Decoder
        self.adt_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, adt_dim),
            nn.ReLU(),
        )

        self.apply(init_weights)

    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def encode(self, x, mode="rna"):
        if mode == "rna":
            latent = self.rna_encoder(x)
        else:
            latent = self.adt_encoder(x)
        mu, sigma = latent[:, : self.latent_dim], torch.exp(
            0.5 * latent[:, self.latent_dim :]
        )
        z = self.reparameterize(mu, sigma)
        return z, mu, sigma

    def decode(self, z, mode="rna"):
        if mode == "rna":
            return self.rna_decoder(z)
        else:
            return self.adt_decoder(z)

    def forward(self, x, mode="rna"):
        if mode == "rna":
            if x.shape[1] != 10000:
                raise ValueError(
                    f"Expected RNA input shape (batch_size, 10000), got {x.shape}"
                )
            z, mu, sigma = self.encode(x, mode="rna")
            shared_z = self.shared_transform(z)
            output = self.decode(shared_z, mode="rna")
        elif mode == "adt":
            if x.shape[1] != 25:
                raise ValueError(
                    f"Expected ADT input shape (batch_size, 25), got {x.shape}"
                )
            z, mu, sigma = self.encode(x, mode="adt")
            shared_z = self.shared_transform(z)
            output = self.decode(shared_z, mode="adt")
        elif mode == "rna_to_adt":
            if x.shape[1] != 10000:
                raise ValueError(
                    f"Expected RNA input shape (batch_size, 10000), got {x.shape}"
                )
            z, mu, sigma = self.encode(x, mode="rna")
            shared_z = self.shared_transform(z)
            output = self.decode(shared_z, mode="adt")
        elif mode == "adt_to_rna":
            if x.shape[1] != 25:
                raise ValueError(
                    f"Expected ADT input shape (batch_size, 25), got {x.shape}"
                )
            z, mu, sigma = self.encode(x, mode="adt")
            shared_z = self.shared_transform(z)
            output = self.decode(shared_z, mode="rna")
        else:
            raise ValueError(f"Invalid mode: {mode}.")
        return output, mu, sigma

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        if self.phase == "rna_to_adt":
            # Here, batch is a tuple: (rna_sample, adt_sample)
            rna_sample, adt_sample = batch
            preds_rna2adt, mu1, sigma1 = self.forward(rna_sample, mode="rna_to_adt")
            loss_rna2adt = self.loss_fn(preds_rna2adt, adt_sample)
            kl_div_adt = -0.5 * torch.sum(
                1 + torch.log(sigma1.pow(2)) - mu1.pow(2) - sigma1.pow(2)
            )
            total_loss = loss_rna2adt + 0.00001 * kl_div_adt
            corr_rna2adt = pearson_corrcoef(preds_rna2adt, adt_sample)
            self.log("rna_to_adt_training_pearson", corr_rna2adt, prog_bar=True, sync_dist=True)
            self.log("rna_to_adt_training_loss", total_loss, sync_dist=True)
            self.log("rna_to_adt_kl_divergence_adt", kl_div_adt, sync_dist=True)
            return total_loss
        elif self.phase == "adt_to_rna":            
            rna_sample, adt_sample = batch
            preds_adt2rna, mu2, sigma2 = self.forward(adt_sample, mode="adt_to_rna")
            loss_adt2rna = self.loss_fn(preds_adt2rna, rna_sample)
            kl_div_rna = -0.5 * torch.sum(
                1 + torch.log(sigma2.pow(2)) - mu2.pow(2) - sigma2.pow(2)
            )
            total_loss = loss_adt2rna + 0.00001 * kl_div_rna
            corr_adt2rna = pearson_corrcoef(preds_adt2rna, rna_sample)
            self.log("adt_to_rna_trainin_pearson", corr_adt2rna, prog_bar=True, sync_dist=True)
            self.log("adt_to_rna_training_loss", total_loss, sync_dist=True)
            self.log("adt_to_rna_kl_divergence_rna", kl_div_rna, sync_dist=True)
            return total_loss
        else:
            # Existing autoencoder / modality-specific logic:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            if self.phase == "rna":
                mode = "rna"
            elif self.phase == "adt":
                mode = "adt"
            else:
                raise ValueError("Invalid training phase specified.")

            x_reconstructed, mu, sigma = self.forward(x, mode=mode)
            reconstruction_loss = self.loss_fn(x_reconstructed, x)
            kl_div = -0.5 * torch.sum(
                1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
            )
            loss = reconstruction_loss + 0.00001 * kl_div
            corr = pearson_corrcoef(x_reconstructed, x)
            self.log(f"{mode}_training_loss", loss, on_step=True, sync_dist=True)
            self.log(f"{mode}_training_reconstruction_loss", reconstruction_loss,  prog_bar=True,  on_step=True, sync_dist=True)
            self.log(f"{mode}_training_pearson_corr", corr, on_step=True, sync_dist=True)
            self.log(f"{mode}_training_kl_divergence", kl_div, sync_dist=True)
            return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.phase == "rna_to_adt":
            if isinstance(batch, (tuple, list)):
                rna_sample, adt_sample = batch
            else:
                raise ValueError("Expected tuple for cross_modal validation")
            preds, mu, sigma = self.forward(rna_sample, mode="rna_to_adt")
            reconstruction_loss = self.loss_fn(preds, adt_sample)
            kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = reconstruction_loss + 0.00001 * kl_div
            corr = pearson_corrcoef(preds, adt_sample)
            self.log("rna_to_adt_validation_loss", loss, prog_bar=True, sync_dist=True)
            self.log("rna_to_adt_validation_reconstruction", reconstruction_loss, sync_dist=True)
            self.log("rna_to_adt_validation_pearson", corr, prog_bar=True, sync_dist=True)
            return loss
        elif self.phase == "adt_to_rna":
            if isinstance(batch, (tuple, list)):
                rna_sample, adt_sample = batch
            else:
                raise ValueError("Expected tuple for cross_modal validation")
            preds, mu, sigma = self.forward(adt_sample, mode="adt_to_rna")
            reconstruction_loss = self.loss_fn(preds, rna_sample)
            kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = reconstruction_loss + 0.00001 * kl_div
            corr = pearson_corrcoef(preds, rna_sample)
            self.log("adt_to_rna_validation_loss", loss, prog_bar=True, sync_dist=True)
            self.log("adt_to_rna_validation_reconstruction_loss", reconstruction_loss, sync_dist=True)
            self.log("adt_to_rna_validation_pearson", corr, prog_bar=True, sync_dist=True)
            return loss
        else:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            target = x
            if self.phase == "rna":
                mode = "rna"
            elif self.phase == "adt":
                mode = "adt"
            else:
                raise ValueError("Invalid validation phase specified.")

        x_reconstructed, mu, sigma = self.forward(x, mode=mode)
        reconstruction_loss = self.loss_fn(x_reconstructed, target)
        kl_div = -0.5 * torch.sum(
            1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
        )
        loss = reconstruction_loss + 0.00001 * kl_div
        self.log(f"{mode}_validation_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{mode}_validation_reconstruction_loss", reconstruction_loss, sync_dist=True)
        corr = pearson_corrcoef(x_reconstructed, target)
        self.log(f"{mode}_validation_pearson_corr", corr, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        rna_sample, adt_sample = batch

        # Forward pass: RNA -> ADT
        pred_adt, _, _ = self.forward(rna_sample, mode="rna_to_adt")
        loss = F.mse_loss(pred_adt, adt_sample)
        corr = pearson_corrcoef(pred_adt, adt_sample)
        r2 = r2_score(pred_adt, adt_sample)

        self.log("rna_to_adt_test_loss", loss)
        self.log("rna_to_adt_test_correlation", corr)
        self.log("rna_to_adt_test_R²", r2)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # For prediction, batch contains test RNA data.
        # We predict ADT from RNA using the "rna_to_adt" mapping.
        x = batch
        preds, _, _ = self.forward(x, mode="rna_to_adt")
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

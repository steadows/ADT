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
import math



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
            
def kl_annealing_factor(schedule, step, total_steps, max_kl=1e-3):
    """Compute KL annealing factor based on the selected schedule."""
    if schedule == "none":
        return 1.0
    elif schedule == "linear":
        return min(max_kl, step / total_steps * max_kl)

    elif schedule == "sigmoid":
        return min(max_kl, max_kl / (1 + math.exp(-5 * (step / total_steps - 0.5))))

    elif schedule == "exponential":
        if total_steps == 0:
            return 0.0  # or some safe default
        return min(max_kl, max_kl * (1 - math.exp(-5 * step / total_steps)))

    else:
        raise ValueError(f"Unknown KL annealing schedule: {schedule}")

class CrossModalVAE(pl.LightningModule):
    def __init__(self, lr, rna_dim=10000, adt_dim=25):
        super().__init__()
        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self.latent_dim = 32
        self.phase = "autoencoder"  # Default phase
        
         # KL and weight hyperparameters:
        self.max_kl = 1e-3  # or another appropriate value
        self.kl_weight_rna = 1e-6
        self.kl_weight_adt = 1e-6
        self.kl_weight_cross = 1e-5
        
        self.cross_modal_losses = []
        self.validation_losses = []
        
        # RNA Encoder
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
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
            nn.Linear(64, self.latent_dim * 2),  # Latent space projection
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
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, rna_dim),
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

    def training_step(self, batch, batch_idx):
        # If the batch is a tuple of length 1 and the first element is a dict, unpack it.
        # Check if batch is a tuple with the first element as a dict.
        if isinstance(batch, tuple):
            if len(batch) > 0 and isinstance(batch[0], dict):
                data = batch[0]
            else:
                raise ValueError(f"Unexpected batch structure: {batch}")
        elif isinstance(batch, dict):
            data = batch
        else:
            raise ValueError(f"Unexpected batch structure: {batch}")

        # Now extract the tensors from the dictionary.
        rna_batch = data.get("rna")
        adt_batch = data.get("adt")
        if rna_batch is None or adt_batch is None:
            raise ValueError("Batch dictionary does not contain 'rna' or 'adt' keys.")
        
        # Compute total training steps dynamically
        total_steps = self.trainer.estimated_stepping_batches

        # Select KL annealing strategy (set this in train.py)
        schedule = self.kl_schedule  # Can be "linear", "sigmoid", or "exponential"
        kl_factor = kl_annealing_factor(schedule, self.global_step, total_steps, self.max_kl)

        if self.phase == "autoencoder":
            
            if batch_idx % 2 == 0:
                # Train RNA autoencoder
                rna_recon, rna_mu, rna_sigma = self.forward(rna_batch, mode="rna")
                rna_recon_loss = self.loss_fn(rna_recon, rna_batch)
                rna_kl_div = -0.5 * torch.sum(1 + torch.log(rna_sigma.pow(2)) - rna_mu.pow(2) - rna_sigma.pow(2))
                loss = rna_recon_loss + (self.kl_weight_rna * kl_factor) * rna_kl_div

                # Logging
                self.log("autoencoder_rna_train_loss", loss, prog_bar=True, batch_size=rna_batch.size(0))
                self.log("autoencoder_rna_train_kl_div", rna_kl_div, batch_size=rna_batch.size(0))
                self.log("autoencoder_rna_recon_train_loss", rna_recon_loss, batch_size=rna_batch.size(0))
                self.log("kl_annealing_factor", kl_factor, batch_size=rna_batch.size(0))
                
                return loss

            else:
                # Train ADT autoencoder
                adt_recon, adt_mu, adt_sigma = self.forward(adt_batch, mode="adt")
                adt_recon_loss = self.loss_fn(adt_recon, adt_batch)
                adt_kl_div = -0.5 * torch.sum(1 + torch.log(adt_sigma.pow(2)) - adt_mu.pow(2) - adt_sigma.pow(2))
                loss = adt_recon_loss + (self.kl_weight_adt * kl_factor) * adt_kl_div

                # Logging
                self.log("autoencoder_adt_train_loss", loss, prog_bar=True, batch_size=rna_batch.size(0))
                self.log("autoencoder_adt_train_kl_div", adt_kl_div, batch_size=rna_batch.size(0))
                self.log("autoencoder_adt_recon_train_loss", adt_recon_loss, batch_size=rna_batch.size(0))
                self.log("kl_annealing_factor", kl_factor, batch_size=rna_batch.size(0))
                
                return loss

        elif self.phase == "cross_modal":
            if batch_idx % 2 == 0:
                # RNA to ADT mapping
                preds_rna2adt, mu_rna, sigma_rna = self.forward(rna_batch, mode="rna_to_adt")
                loss_rna2adt = self.loss_fn(preds_rna2adt, adt_batch)
                kl_div_rna2adt = -0.5 * torch.sum(1 + torch.log(sigma_rna.pow(2)) - mu_rna.pow(2) - sigma_rna.pow(2))
                loss = loss_rna2adt + (self.kl_weight_cross * kl_factor) * kl_div_rna2adt

                # Logging
                self.log("crossmodal_rna2adt_train_loss", loss, prog_bar=True, batch_size=rna_batch.size(0))
                self.log("crossmodal_rna2adt_train_kl_div", kl_div_rna2adt, batch_size=rna_batch.size(0))
                self.log("crossmodal_rna2adt_recon_train_loss", loss_rna2adt, batch_size=rna_batch.size(0))
                self.log("kl_annealing_factor", kl_factor, batch_size=rna_batch.size(0))
                
                return loss

            else:
                # ADT to RNA mapping
                preds_adt2rna, mu_adt, sigma_adt = self.forward(adt_batch, mode="adt_to_rna")
                loss_adt2rna = self.loss_fn(preds_adt2rna, rna_batch)
                kl_div_adt2rna = -0.5 * torch.sum(1 + torch.log(sigma_adt.pow(2)) - mu_adt.pow(2) - sigma_adt.pow(2))
                loss = loss_adt2rna + (self.kl_weight_cross * kl_factor) * kl_div_adt2rna

                # Logging
                self.log("crossmodal_adt2rna_train_loss", loss, prog_bar=True, batch_size=rna_batch.size(0))
                self.log("crossmodal_adt2rna_train_kl_div", kl_div_adt2rna, batch_size=rna_batch.size(0))
                self.log("crossmodal_adt2rna_recon_train_loss", loss_adt2rna, batch_size=rna_batch.size(0))
                self.log("kl_annealing_factor", kl_factor, batch_size=rna_batch.size(0))

                return loss

    def validation_step(self, batch, batch_idx):
        # Check if batch is a tuple with the first element as a dict.
        if isinstance(batch, tuple):
            if len(batch) > 0 and isinstance(batch[0], dict):
                data = batch[0]
            else:
                raise ValueError(f"Unexpected batch structure: {batch}")
        elif isinstance(batch, dict):
            data = batch
        else:
            raise ValueError(f"Unexpected batch structure: {batch}")

        # Now extract the tensors from the dictionary.
        rna_batch = data.get("rna")
        adt_batch = data.get("adt")
        if rna_batch is None or adt_batch is None:
            raise ValueError("Batch dictionary does not contain 'rna' or 'adt' keys.")

        if self.phase == "autoencoder":
            if batch_idx % 2 == 0:
                # RNA Autoencoder validation
                rna_recon, rna_mu, rna_sigma = self.forward(rna_batch, mode="rna")
                rna_recon_loss = self.loss_fn(rna_recon, rna_batch)
                rna_kl_div = -0.5 * torch.sum(1 + torch.log(rna_sigma.pow(2)) - rna_mu.pow(2) - rna_sigma.pow(2))
                rna_total_loss = rna_recon_loss + self.kl_weight_rna * rna_kl_div
                rna_corr = pearson_corrcoef(rna_recon, rna_batch)

                # Logging
                self.log("autoencoder_rna_val_loss", rna_total_loss, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("autoencoder_rna_val_recon_loss", rna_recon_loss, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("autoencoder_rna_val_kl_div", rna_kl_div, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("autoencoder_rna_val_pearson", rna_corr, prog_bar=True, sync_dist=True, batch_size=rna_batch.size(0))
                loss = rna_total_loss
                self.validation_losses.append(loss.detach())
                return

            else:
                # ADT Autoencoder validation
                adt_recon, adt_mu, adt_sigma = self.forward(adt_batch, mode="adt")
                adt_recon_loss = self.loss_fn(adt_recon, adt_batch)
                adt_kl_div = -0.5 * torch.sum(1 + torch.log(adt_sigma.pow(2)) - adt_mu.pow(2) - adt_sigma.pow(2))
                adt_total_loss = adt_recon_loss + self.kl_weight_adt * adt_kl_div
                adt_corr = pearson_corrcoef(adt_recon, adt_batch)
                

                # Logging
                self.log("autoencoder_adt_val_loss", adt_total_loss, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("autoencoder_adt_val_recon_loss", adt_recon_loss, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("autoencoder_adt_val_kl_div", adt_kl_div, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("autoencoder_adt_val_pearson", adt_corr, prog_bar=True, sync_dist=True, batch_size=rna_batch.size(0))
                loss = adt_total_loss
                self.validation_losses.append(loss.detach())
                return
            

        elif self.phase == "cross_modal":
            if batch_idx % 2 == 0:
                # RNA to ADT validation
                preds_rna2adt, mu_rna, sigma_rna = self.forward(rna_batch, mode="rna_to_adt")
                loss_rna2adt = self.loss_fn(preds_rna2adt, adt_batch)
                kl_div_rna2adt = -0.5 * torch.sum(1 + torch.log(sigma_rna.pow(2)) - mu_rna.pow(2) - sigma_rna.pow(2))
                loss = loss_rna2adt + self.kl_weight_cross * kl_div_rna2adt
                corr_rna2adt = pearson_corrcoef(preds_rna2adt, adt_batch)

                # Logging
                self.log("crossmodal_rna2adt_val_loss", loss, prog_bar=True, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("crossmodal_rna2adt_val_recon_loss", loss_rna2adt, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("crossmodal_rna2adt_val_kl_div", kl_div_rna2adt, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("crossmodal_rna2adt_val_pearson", corr_rna2adt, prog_bar=True, sync_dist=True, batch_size=rna_batch.size(0))
                
                self.cross_modal_losses.append(loss.detach())
                return

            else:
                # ADT to RNA validation
                preds_adt2rna, mu_adt, sigma_adt = self.forward(adt_batch, mode="adt_to_rna")
                loss_adt2rna = self.loss_fn(preds_adt2rna, rna_batch)
                kl_div_adt2rna = -0.5 * torch.sum(1 + torch.log(sigma_adt.pow(2)) - mu_adt.pow(2) - sigma_adt.pow(2))
                loss = loss_adt2rna + self.kl_weight_cross * kl_div_adt2rna
                corr_adt2rna = pearson_corrcoef(preds_adt2rna, rna_batch)

                # Logging
                self.log("crossmodal_adt2rna_val_loss", loss, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("crossmodal_adt2rna_val_recon_loss", loss_adt2rna, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("crossmodal_adt2rna_val_kl_div", kl_div_adt2rna, sync_dist=True, batch_size=rna_batch.size(0))
                self.log("crossmodal_adt2rna_val_pearson", corr_adt2rna, sync_dist=True, batch_size=rna_batch.size(0))
                
                self.cross_modal_losses.append(loss.detach())
                return
        


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
    
    def on_validation_epoch_end(self):
        if self.validation_losses:
            combined_loss = torch.stack(self.validation_losses).mean()
            self.log("combined_val_loss", combined_loss, prog_bar=True, sync_dist=True)
            self.validation_losses = [] 
            
        # For cross-modal, if you've been accumulating losses in self.cross_modal_losses:
        if hasattr(self, "cross_modal_losses") and self.cross_modal_losses:
            combined_cross_modal_loss = torch.stack(self.cross_modal_losses).mean()
            self.log("crossmodal_combined_val_loss", combined_cross_modal_loss, prog_bar=True, sync_dist=True)
            # Clear for next epoch
            self.cross_modal_losses = []


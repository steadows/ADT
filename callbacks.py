# train.py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import pytorch_lightning as pl

# Define your custom callback here
class SharedEmbeddingCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        from dataset import PairedEvalDataset  # make sure this is imported at the top
        # Use the evaluation tensors from your DataModule
        dataset = PairedEvalDataset(trainer.datamodule.eval_rna, trainer.datamodule.eval_adt)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=trainer.datamodule.batch_size, shuffle=False)
        batch = next(iter(val_loader))
        # Assume batch is a tuple: (rna_samples, adt_samples)
        rna_samples, adt_samples = batch
        
        # Move to the correct device
        rna_samples = rna_samples.to(pl_module.device)
        adt_samples = adt_samples.to(pl_module.device)
        
        # Get latent representations from both modalities
        rna_latent, _, _ = pl_module.encode(rna_samples, mode="rna")
        adt_latent, _, _ = pl_module.encode(adt_samples, mode="adt")
        
        # Concatenate them into one embedding tensor
        combined_embedding = torch.cat([rna_latent, adt_latent], dim=0)
        
        # Create combined metadata. Prefix each name with modality info.
        rna_metadata = [f"RNA_{name}" for name in trainer.datamodule.val_rna_sample_names[:rna_latent.size(0)]]
        adt_metadata = [f"ADT_{name}" for name in trainer.datamodule.val_adt_sample_names[:adt_latent.size(0)]]
        combined_metadata = rna_metadata + adt_metadata
        
        # Use a global step that updates each epoch (or current_epoch)
        global_step = trainer.current_epoch
        
        # Log the combined embeddings with a single tag
        trainer.logger.experiment.add_embedding(
            combined_embedding,
            metadata=combined_metadata,
            global_step=global_step,
            tag="shared_latent_space"
        )
import torch
import pandas as pd
import pytorch_lightning as pl
from model import CrossModalVAE
from data_module import RNAADTDataModule

def main():
    checkpoint_path = "checkpoints/best_model_rna_to_adt.ckpt"  # Ensure the checkpoint path is correct
    model = CrossModalVAE.load_from_checkpoint(checkpoint_path)
    model.phase = "rna_to_adt"
    model.eval()
    
    # Move model to device (MPS if available, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    # Setup data module for testing (submission stage)
    dm = RNAADTDataModule(batch_size=64, num_workers=4, data_dir="data/")
    dm.setup(stage="test")
    
    # Use submission_dataloader to load only test RNA data
    submission_loader = dm.submission_dataloader()
    
    # Use Lightning's predict() function
    trainer = pl.Trainer(accelerator="cpu", devices=1)
    predictions = trainer.predict(model, dataloaders=submission_loader)
    
    # Concatenate predictions (predictions is a list of tensors)
    all_predictions = torch.cat(predictions, dim=0)
    pred_array = all_predictions.numpy()  # Expected shape: [5000, 25] if test_rna has 5000 samples
    num_samples, num_features = pred_array.shape
    long_predictions = pred_array.flatten()  # Row-major flattening
    
    # Generate IDs in the format "id1", "id2", ...
    ids = [f"id{i+1}" for i in range(num_samples * num_features)]
    
    # Create submission DataFrame and save to CSV
    submission_df = pd.DataFrame({
        "Id": ids,
        "Expected": long_predictions
    })
    submission_df.to_csv("submission_1.csv", index=False)
    print("Submission file saved as submission.csv")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Useful on Windows
    main()

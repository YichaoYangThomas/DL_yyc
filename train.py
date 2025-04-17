import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from tqdm import tqdm
import random
import numpy as np
from dataset import create_wall_dataloader
from models import JEPAModel


def compute_cosine_loss(pred_repr, target_repr):
    """Compute loss based on cosine similarity between predictions and targets"""
    similarities = F.cosine_similarity(pred_repr, target_repr, dim=-1)
    loss_per_sample = 1 - similarities
    loss_per_sample = loss_per_sample.sum(dim=1)
    return loss_per_sample.mean()


def compute_barlow_twins_loss(z1, z2, lambda_coef=0.005):
    """Compute Barlow Twins loss to prevent representation collapse"""
    # Get dimensions and reshape
    batch_size, time_steps, feat_dim = z1.shape
    z1_flat = z1.view(-1, feat_dim)
    z2_flat = z2.view(-1, feat_dim)
    
    # Normalize embeddings along batch dimension
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)
    
    # Cross-correlation matrix
    batch_size = z1_norm.shape[0]
    cross_corr = torch.matmul(z1_norm.T, z2_norm) / batch_size
    
    # Separate diagonal and off-diagonal terms
    diag_loss = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
    off_diag = cross_corr.flatten()[:-1].view(cross_corr.size(0) - 1, cross_corr.size(1) + 1)[:, 1:].pow_(2)
    off_diag_loss = off_diag.sum()
    
    # Combined loss
    total_loss = diag_loss + lambda_coef * off_diag_loss
    return total_loss


def train_model(device):
    """Train the JEPA model"""
    # Dataset path configuration
    data_path = "/scratch/DL25SP/train"
    print(f"Loading training data from: {data_path}")
    
    # Create data loader
    train_loader = create_wall_dataloader(
        data_path=data_path,
        probing=False,
        device=device,
        train=True,
    )
    
    # Initialize model
    print("Initializing model architecture...")
    model = JEPAModel(device=device).to(device)
    
    # Training hyperparameters
    num_epochs = 10
    learning_rate = 1e-4
    jepa_weight = 0.2
    
    # Model summary
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {param_count:,}")
    
    # Configure optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    # Main training loop
    for epoch in range(num_epochs):
        # Set training mode
        model.train()
        epoch_loss = 0.0
        
        # Progress tracking
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Batch processing
        for batch_idx, batch in enumerate(progress_bar):
            # Get batch data
            obs_states = batch.states
            act_seq = batch.actions
            
            # Forward pass
            pred_states = model(obs_states, act_seq)
            
            # Compute targets (without gradient)
            with torch.no_grad():
                target_states = model.encoder(
                    obs_states.view(-1, *obs_states.shape[2:])
                ).view(obs_states.size(0), obs_states.size(1), -1)
            
            # Compute primary JEPA loss
            repr_loss = F.smooth_l1_loss(pred_states, target_states)
            
            # Compute representations for Barlow Twins loss
            encoded_states = model.encoder(
                obs_states.view(-1, *obs_states.shape[2:])
            ).view(obs_states.size(0), obs_states.size(1), -1)
            
            # Compute regularization loss
            bt_loss = compute_barlow_twins_loss(encoded_states, pred_states)
            
            # Combined loss
            batch_loss = jepa_weight * repr_loss + (1 - jepa_weight) * bt_loss
            
            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += batch_loss.item()
            
            # Update progress display
            progress_bar.set_postfix({
                'loss': f'{batch_loss.item():.4f}'
            })
        
        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed, avg_loss: {avg_loss:.8e}")
    
    # Save trained model
    print("Saving model checkpoint...")
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Training complete!")
    
    return model


def get_device():
    """Determine and return the appropriate computation device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
    
    # Get computation device
    device = get_device()
    
    # Train model
    model = train_model(device)
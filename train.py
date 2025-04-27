import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from tqdm import tqdm
import random
import numpy as np
from dataset import create_wall_dataloader
from models import JEPAModel


def je_loss(predictions, targets):
    """Compute cosine similarity based loss"""
    similarities = F.cosine_similarity(predictions, targets, dim=-1)
    loss_per_sample = 1 - similarities
    loss_per_sample = loss_per_sample.sum(dim=1)
    return loss_per_sample.mean()


def barlow_twins_loss(z1, z2, lambda_param=0.005):
    """Compute Barlow Twins loss to prevent representation collapse"""
    # Get dimensions and flatten batch
    B, T, D = z1.shape
    z1_flat = z1.view(-1, D)
    z2_flat = z2.view(-1, D)
    
    # Normalize representations
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)
    
    # Compute correlation matrix
    batch_size = z1_norm.shape[0]
    corr_matrix = torch.matmul(z1_norm.T, z2_norm) / batch_size
    
    # Compute diagonal and off-diagonal losses
    diag_loss = torch.diagonal(corr_matrix).add_(-1).pow_(2).sum()
    off_diag_loss = corr_matrix.flatten()[:-1].view(corr_matrix.size(0) - 1, corr_matrix.size(1) + 1)[:, 1:].pow_(2).sum()
    
    # Combined loss
    return diag_loss + lambda_param * off_diag_loss


def train_model(device):
    """Train the JEPA model"""
    # Data path
    data_path = "/scratch/DL25SP/train"
    print(f"Loading training data... Path: {data_path}")
    
    # Create data loader
    train_loader = create_wall_dataloader(
        data_path=data_path,
        probing=False,
        device=device,
        train=True,
    )
    
    # Initialize model
    print("Initializing model...")
    model = JEPAModel(device=device).to(device)
    
    # Training configuration
    num_epochs = 200
    learning_rate = 5e-5
    jepa_loss_weight = 0.2
    
    # Count model parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Process each batch
        for batch_idx, batch in enumerate(progress_bar):
            states = batch.states
            actions = batch.actions
            
            # Forward pass - get predicted representations
            predictions = model(states, actions)
            
            # Get target representations (no gradient)
            with torch.no_grad():
                targets = model.encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)
            
            # Compute JEPA loss using smooth L1 loss
            jepa_loss = F.smooth_l1_loss(predictions, targets)
            
            # Get original representations for Barlow Twins loss
            z1 = model.encoder(
                states.view(-1, *states.shape[2:])
            ).view(states.size(0), states.size(1), -1)
            z2 = predictions
            
            # Compute Barlow Twins loss
            bt_loss = barlow_twins_loss(z1, z2)
            
            # Combined loss with weighting
            batch_loss = jepa_loss_weight * jepa_loss + (1 - jepa_loss_weight) * bt_loss
            
            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            # Record loss
            total_loss += batch_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed, average loss: {avg_loss:.8e}")
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Training completed!")
    
    return model


def get_device():
    """Get available compute device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


if __name__ == "__main__":
    # Set random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
    
    # Get device
    device = get_device()
    
    # Start training
    model = train_model(device)
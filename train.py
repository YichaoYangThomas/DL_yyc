import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from tqdm import tqdm
import random
import numpy as np
from dataset import create_wall_dataloader
from models import JEPAModel


def barlow_twins_loss(z1, z2, lambda_param=0.005):
    """Barlow Twins loss calculation"""
    # Get dimensions
    B, T, D = z1.shape
    z1_flat = z1.view(-1, D)
    z2_flat = z2.view(-1, D)
    
    # Normalize
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)
    
    # Correlation matrix
    batch_size = z1_norm.shape[0]
    c = torch.matmul(z1_norm.T, z2_norm) / batch_size
    
    # Loss components
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[:-1].view(c.size(0) - 1, c.size(1) + 1)[:, 1:].pow_(2).sum()
    
    # Combined loss
    loss = on_diag + lambda_param * off_diag
    return loss


def apply_augmentation(states):
    """Apply data augmentation to states"""
    B, T, C, H, W = states.shape
    aug_states = []
    
    for b in range(B):
        sample = states[b]
        
        # Random transformations
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5
        angle = random.choice([0, 90, 180, 270])
        
        # Process each frame
        aug_frames = []
        for t in range(T):
            frame = sample[t]
            
            # Apply transformations
            if do_hflip:
                frame = VF.hflip(frame)
            if do_vflip:
                frame = VF.vflip(frame)
            frame = VF.rotate(frame, angle)
            
            # Add noise
            noise = torch.randn_like(frame) * 0.01
            frame = frame + noise
            
            aug_frames.append(frame)
        
        # Stack time dimension
        aug_frames = torch.stack(aug_frames, dim=0)
        aug_states.append(aug_frames)
    
    # Stack batch dimension
    result = torch.stack(aug_states, dim=0)
    return result


def train_model(device):
    """Train the JEPA model"""
    print("Loading training data...")
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL25SP/train",
        probing=False,
        device=device,
        train=True,
    )
    
    print("Initializing model...")
    model = JEPAModel(device=device).to(device)
    
    # Training settings
    num_epochs = 10
    jep_co = 0.2
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get batch data
            states = batch.states
            actions = batch.actions
            
            # Data augmentation
            states_aug = apply_augmentation(states)
            
            # Forward pass
            predictions = model(states_aug, actions)
            
            # Get target representations
            with torch.no_grad():
                targets = model.encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)
            
            # JEPA loss
            jepa_loss = F.smooth_l1_loss(predictions, targets)
            
            # Get encoded representations for Barlow Twins loss
            z1 = model.encoder(
                states.view(-1, *states.shape[2:])
            ).view(states.size(0), states.size(1), -1)
            z2 = predictions
            bt_loss = barlow_twins_loss(z1, z2)
            
            # Total loss
            total_loss_batch = jep_co * jepa_loss + (1-jep_co) * bt_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            # Track loss
            total_loss += total_loss_batch.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}',
                'jepa': f'{jepa_loss.item():.4f}',
                'barlow': f'{bt_loss.item():.4f}'
            })
        
        # Epoch summary
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8e}")
    
    # Save model
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Training complete!")
    return model


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


if __name__ == "__main__":
    # Set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Get device
    device = get_device()
    
    # Train model
    model = train_model(device)
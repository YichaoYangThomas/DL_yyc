import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from tqdm import tqdm
import random
from dataset import create_wall_dataloader
from models import JEPAModel


def je_loss(predictions, targets):
    similarities = F.cosine_similarity(predictions, targets, dim=-1)
    loss_per_sample = 1 - similarities
    loss_per_sample = loss_per_sample.sum(dim=1)
    loss = loss_per_sample.mean()
    return loss


def barlow_twins_loss(z1, z2, lambda_param=0.005):
    # b, t, d
    B, T, D = z1.shape
    z1_flat = z1.view(-1, D)
    z2_flat = z2.view(-1, D)
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)
    batch_size = z1_norm.shape[0]
    c = torch.matmul(z1_norm.T, z2_norm) / batch_size
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[:-1].view(c.size(0) - 1, c.size(1) + 1)[:, 1:].pow_(2).sum()
    loss = on_diag + lambda_param * off_diag
    return loss


def apply_augmentation(states):
    B, T, C, H, W = states.shape
    aug_states = []
    for b in range(B):
        sample = states[b]
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        aug_frames = []
        for t in range(T):
            frame = sample[t]
            if do_hflip:
                frame = VF.hflip(frame)
            if do_vflip:
                frame = VF.vflip(frame)
            frame = VF.rotate(frame, angle)
            noise = torch.randn_like(frame) * 0.01
            frame = frame + noise
            aug_frames.append(frame)
        aug_frames = torch.stack(aug_frames, dim=0)
        aug_states.append(aug_frames)
    aug_states = torch.stack(aug_states, dim=0)
    return aug_states


def train_model(device):
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        train=True,
    )

    model = JEPAModel(device=device).to(device)

    num_epochs = 10

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            states = batch.states
            actions = batch.actions
            # temporal states rep s0 s1 tilde ...
            predictions = model(states, actions)
            # temporal states rep s0 s1' ...
            with torch.no_grad():
                targets = model.encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)
            # distance
            jepa_loss = F.smooth_l1_loss(predictions, targets)

            # with grad
            # temporal states rep s0 s1' ...
            z1 = model.encoder(
                states.view(-1, *states.shape[2:])
            ).view(states.size(0), states.size(1), -1)
            # temporal states rep s0 s1 tilde ...
            z2 = predictions
            # barlow twins loss between predicted and encoded reps
            bt_loss = barlow_twins_loss(z1, z2)

            # back
            jep_co = 0.2
            total_loss_batch = jep_co * jepa_loss + (1-jep_co) * bt_loss
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            total_loss += total_loss_batch.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8e}")
    # Save
    torch.save(model.state_dict(), 'model_weights.pth')


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


if __name__ == "__main__":
    device = get_device()
    model = train_model(device)

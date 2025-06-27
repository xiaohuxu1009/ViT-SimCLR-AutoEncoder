import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import SimCLRDualViewDataset
from model import get_vit_encoder, SimCLR
from loss import nt_xent_loss
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp import autocast, GradScaler


def train_simclr(train_root, val_root, epochs=500, batch_size=256, lr=1e-3, temperature=0.07, device='cuda'):
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomResizedCrop(128, scale=(0.5, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)],
            p=0.8,
        ),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.2, 0.8))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SimCLRDualViewDataset(root=train_root, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = SimCLRDualViewDataset(root=val_root, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    encoder = get_vit_encoder()
    model = SimCLR(encoder, projection_dim=128).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=epochs,
        lr_min=1e-6,
        warmup_t=10,
        warmup_lr_init=lr * 0.1,
        cycle_limit=1,
        t_in_epochs=True
    )

    scaler = GradScaler()

    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (xi, xj) in enumerate(train_loader):
            xi = xi.to(device)
            xj = xj.to(device)
            optimizer.zero_grad()

            with autocast():
                hi, zi = model(xi)
                hj, zj = model(xj)
                loss = nt_xent_loss(zi, zj, temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_samples += 1

        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples

        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for xi, xj in val_loader:
                xi = xi.to(device)
                xj = xj.to(device)
                with autocast():
                    _, zi = model(xi)
                    _, zj = model(xj)
                    loss = nt_xent_loss(zi, zj, temperature)
                val_loss += loss.item()
                val_samples += 1
        avg_val_loss = val_loss / val_samples
        model.train()

        print(
            f"Epoch [{epoch + 1}/{epochs}] Train Loss: {avg_loss:.4f}  Val Loss: {avg_val_loss:.4f}  Time: {epoch_time:.2f}s")

        scheduler.step(epoch + 1)

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'simclr_model_epoch_{epoch + 1}.pth')
            print(f"Model saved: simclr_model_epoch_{epoch + 1}.pth")

    return model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_folder = './dataset/train'
    val_folder = './dataset/val'
    print("Start training SimCLR……")
    model = train_simclr(train_folder, val_folder, epochs=500, batch_size=256, lr=1e-3, temperature=0.07, device=device)
    torch.save(model.state_dict(), 'simclr_model_final.pth')
    print("Finish training, final model saved: simclr_model_final.pth")

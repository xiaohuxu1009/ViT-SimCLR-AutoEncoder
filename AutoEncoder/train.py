import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_autoencoder(model, train_data, val_data=None, epochs=100, lr=1e-3, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_data is not None:
        val_tensor = torch.tensor(val_data, dtype=torch.float32)
        val_dataset = TensorDataset(val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_recon, _ = model(batch)
            loss = criterion(x_recon, batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.size(0)
        avg_train_loss = total_train_loss / len(train_dataset)

        if val_data is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    x_recon, _ = model(batch)
                    loss = criterion(x_recon, batch)
                    total_val_loss += loss.item() * batch.size(0)
            avg_val_loss = total_val_loss / len(val_dataset)
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.6f}")

    return model

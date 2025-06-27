import torch

def extract_features(model, data):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        _, z = model(data_tensor)
        return z.cpu().numpy()

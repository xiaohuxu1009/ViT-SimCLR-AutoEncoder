import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from model import get_vit_encoder, SimCLR
from data import TestDataset

from torch.cuda.amp import autocast

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_folder = './dataset/val/RE_Images'
model_path = 'simclr_model_final.pth'
output_csv = 'features_RE_test.csv'

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = TestDataset(root=image_folder, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load ViT backbone model
encoder = get_vit_encoder()
model = SimCLR(encoder, projection_dim=128).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Extract features
features_list = []

with torch.no_grad():
    for img, img_path in test_loader:
        img = img.to(device)
        with autocast():
            h, _ = model(img)
        h = h.cpu().numpy()
        for i in range(h.shape[0]):
            features_list.append([img_path[i]] + list(h[i]))

if features_list:
    feature_dim = len(features_list[0]) - 1
    columns = ['image_path'] + [f'feature_{i}' for i in range(feature_dim)]
    df = pd.DataFrame(features_list, columns=columns)

    df.to_csv(output_csv, index=False)
    print(f"Features are saved to {output_csv}")
else:
    print("No features extracted. Check if test_loader has data.")

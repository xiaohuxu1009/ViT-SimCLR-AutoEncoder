import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import re

class SimCLRDualViewDataset(Dataset):
    def __init__(self, root, transform):
        self.dataset = datasets.ImageFolder(root=root)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        xi = self.transform(img)
        xj = self.transform(img)
        return xi, xj

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

class TestDataset(Dataset):
    def __init__(self, root, transform):
        self.image_paths = [os.path.join(root, file)
                            for root, _, files in os.walk(root)
                            for file in files if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
        self.image_paths = sorted(self.image_paths, key=natural_sort_key)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, path
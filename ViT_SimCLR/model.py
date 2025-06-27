import torch
import torch.nn as nn
from torchvision import models
import timm

class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder

        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        # h = self.fc(h)
        z = self.projection(h)
        return h, z


def get_vit_encoder(img_size=128):
    vit = timm.create_model('vit_base_patch16_224', img_size=img_size, pretrained=True)
    vit.head = nn.Identity()
    return vit


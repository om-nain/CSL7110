import torch
import torch.nn as nn
import torch.nn.functional as F

class SongEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(SongEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x):
        return self.net(x) 
import torch.nn as nn

class CustomFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=5, stride=5)  # 245 → 49

    def forward(self, x):  # x: [B, 1, 245]
        return self.conv(x)  # → [B, 1, 49]
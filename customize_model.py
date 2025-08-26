import torch.nn as nn

class CustomFeatureExtractor(nn.Module):
    def __init__(self, arch='conv1d'):
        super().__init__()
        self.arch = arch.lower()
        self.extractor = self._build_extractor()

    def _build_extractor(self):
        if self.arch == 'conv1d':
            return nn.Conv1d(1, 512, kernel_size=5, stride=5)

        if self.arch == '2_conv1d_relu':
            return nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=5, stride=2),
                nn.ReLU()
            )

        elif self.arch == 'cnn': # maybe for multi-channel data
            return nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3),
                nn.ReLU()
            )
        elif self.arch == 'mlp':
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(245, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")

    def forward(self, x):
        return self.extractor(x)



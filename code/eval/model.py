"""
Model components for SpectralFM evaluation.
No fairseq dependency.
"""
import torch.nn as nn


SUPPORTED_ARCHS = ["conv1d", "2_conv1d_relu", "2_conv1d", "3_conv1d_relu", "3_conv1d"]


class _TransposeLast(nn.Module):
    def forward(self, x):
        return x.transpose(-2, -1)


def _fairseq_conv_block(n_in: int, n_out: int, k: int, stride: int) -> nn.Sequential:
    """
    One conv block matching fairseq ConvFeatureExtractionModel(mode='layer_norm').
    Keys: [0]=Conv1d, [1]=Dropout, [2]=Sequential([2.0]=TransposeLast, [2.1]=LayerNorm, [2.2]=TransposeLast), [3]=GELU
    """
    return nn.Sequential(
        nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
        nn.Dropout(p=0.0),
        nn.Sequential(
            _TransposeLast(),
            nn.LayerNorm(n_out),
            _TransposeLast(),
        ),
        nn.GELU(),
    )


class FairseqConvFeatureExtractor(nn.Module):
    """
    Multi-layer conv+LayerNorm feature extractor matching a fairseq ConvFeatureExtractionModel
    checkpoint. Key structure matches the checkpoint so weights load with no remapping.
    """

    def __init__(self, layers_config: list | None = None):
        super().__init__()
        if layers_config is None:
            # Default: matches SpectralFM training config
            layers_config = [(512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 5, 5)]
        blocks = []
        in_d = 1
        for out_d, k, stride in layers_config:
            blocks.append(_fairseq_conv_block(in_d, out_d, k, stride))
            in_d = out_d
        self.conv_layers = nn.ModuleList(blocks)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, L] → [B, 1, L]
        for conv in self.conv_layers:
            x = conv(x)
        return x  # [B, C, T]


class CustomFeatureExtractor(nn.Module):
    """Custom 1D-CNN frontend that replaces data2vec's original conv stack."""

    def __init__(self, arch: str = "conv1d"):
        super().__init__()
        self.arch = arch.lower()
        self.extractor = self._build()

    def _build(self):
        a = self.arch
        if a == "conv1d":
            return nn.Conv1d(1, 512, kernel_size=5, stride=5)

        elif a == "2_conv1d_relu":
            return nn.Sequential(
                nn.Conv1d(1, 256, kernel_size=3, stride=3),
                nn.ReLU(),
                nn.Conv1d(256, 512, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.ZeroPad1d((0, 10)),
            )

        elif a == "2_conv1d":
            return nn.Sequential(
                nn.Conv1d(1, 256, kernel_size=3, stride=3),
                nn.Conv1d(256, 512, kernel_size=3, stride=2),
                nn.ZeroPad1d((0, 10)),
            )

        elif a == "3_conv1d_relu":
            return nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(256, 512, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.ZeroPad1d((0, 20)),
            )

        elif a == "3_conv1d":
            return nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=2),
                nn.Conv1d(128, 256, kernel_size=3, stride=2),
                nn.Conv1d(256, 512, kernel_size=3, stride=2),
                nn.ZeroPad1d((0, 20)),
            )

        raise ValueError(f"Unknown arch: {self.arch!r}. Supported: {SUPPORTED_ARCHS}")

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, L] → [B, 1, L]
        return self.extractor(x)


class CompletionHead(nn.Module):
    """Linear head for signal completion (masked prediction) evaluation."""

    def __init__(self, hidden_dim: int = 768, output_dim: int = 245):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def build_model(hf_model, arch: str = "conv1d", with_completion_head: bool = False):
    """
    Attach custom components to a HuggingFace Data2VecAudioModel.
    Returns the modified model (in-place mutation + return).
    """
    hf_model.feature_extractor = CustomFeatureExtractor(arch)
    if with_completion_head:
        hf_model.completion_head = CompletionHead()

    # Freeze all except the custom feature extractor
    for param in hf_model.parameters():
        param.requires_grad = False
    for param in hf_model.feature_extractor.parameters():
        param.requires_grad = True

    return hf_model


def has_completion_head(model) -> bool:
    return hasattr(model, "completion_head")

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq import checkpoint_utils
from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
)
from fairseq.modules import (
    GradMultiply,
    LayerNorm,
)
from fairseq.utils import index_put


logger = logging.getLogger(__name__)


@dataclass
class Data2VecAudioConfig(Wav2Vec2Config):

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )
    train_only_fe: bool = field(
        default=True, metadata={"help": "Train only feature-extractor, freeze other parts"}
    )

    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    model_path: Optional[str] = field(default=None)
    skip_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("data2vec_audio", dataclass=Data2VecAudioConfig)
class Data2VecAudioModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecAudioConfig):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.extractor_embed = feature_enc_layers[-1][0]

        self.model_path = cfg.model_path
        if self.model_path:
            print(f"model_path = { self.model_path }")
        self.ema = None
        self.embed = cfg.encoder_embed_dim

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = nn.Linear(self.extractor_embed, cfg.encoder_embed_dim)

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.extractor_embed)

        self.final_proj = nn.Linear(self.embed, self.embed)

        self.num_updates = 0
        
        # For sanity testing: fixed mask indices range
        self._fixed_mask_start = None
        self._fixed_mask_end = None

        if cfg.train_only_fe:
            self.freeze_all_except_feature_extractor()
        else:
            print("[+] Training the entire model (not only Feature-Extractor)")
            logger.info("[+] Training the entire model (not only Feature-Extractor)")

    def freeze_all_except_feature_extractor(self):
        for name, p in self.named_parameters():
            if "feature_extractor" not in name:
                p.requires_grad = False

        print("[INFO] ONLY feature extractor is trainable")
        logger.info("[INFO] ONLY feature extractor is trainable")

    def load_model_weights(self, state, model, cfg):
        if "_ema" in state["model"]:
            del state["model"]["_ema"]
        model.load_state_dict(state["model"], strict=True)
        print("Loaded model weights")

    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")

        self.ema = EMAModule(
            self.encoder if self.cfg.ema_transformer_only else self,
            ema_config,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is None and self.final_proj is not None:
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecAudioConfig, task=None):
        """Build a new model instance."""
        model = cls(cfg)
        if cfg.model_path and not cfg.skip_pretrained_weights:
                model_path = cfg.model_path
                logger.info(f"Loading pretrained checkpoint from {model_path}")
                print(f"Loading pretrained checkpoint from {model_path}")
                state = checkpoint_utils.load_checkpoint_to_cpu(model_path, {})
                ckpt_model = state.get("model", state)

                missing, unexpected = model.load_state_dict(ckpt_model, strict=False)
                logger.info(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
                print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
                try:
                    # Example for checking a core transformer layer
                    layer_name = "encoder.layers.0.self_attn.k_proj.weight"
                    if layer_name in model.state_dict():
                        weights = model.state_dict()[layer_name]
                        print(f"\n✅ VERIFICATION: Loaded Weights Check for {layer_name}")
                        print(f"  Mean: {weights.float().mean().item():.6f}")
                        print(f"  Std Dev: {weights.float().std().item():.6f}")
                        print(f"  Slice: {weights.flatten()[:5].tolist()}")
                except Exception as e:
                    print(f"Could not inspect weights: {e}")

        return model

        # return cls(cfg)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                # Check if fixed_mask_indices is set (for sanity testing)
                if hasattr(self, '_fixed_mask_start') and hasattr(self, '_fixed_mask_end'):
                    # Use fixed mask range to create mask indices
                    # Import here to avoid circular dependencies
                    import sys
                    import os
                    # Path from fairseq/examples/data2vec/models/ to code/
                    # Go up: models -> data2vec -> examples -> fairseq -> SpectralFM -> code
                    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
                    code_path = os.path.join(base_path, "code")
                    if code_path not in sys.path and os.path.exists(code_path):
                        sys.path.insert(0, code_path)
                    try:
                        from test_utils import create_fixed_mask_indices
                        fixed_mask = create_fixed_mask_indices(
                            B, T, self._fixed_mask_start, self._fixed_mask_end
                        )
                        mask_indices = torch.from_numpy(fixed_mask).to(x.device)
                    except ImportError:
                        # Fallback: create mask directly if import fails
                        import numpy as np
                        mask_start = self._fixed_mask_start
                        mask_end = self._fixed_mask_end
                        if T <= mask_start:
                            fixed_mask = np.zeros((B, T), dtype=bool)
                        else:
                            actual_mask_end = min(mask_end, T - 1)
                            if actual_mask_end < mask_start:
                                fixed_mask = np.zeros((B, T), dtype=bool)
                            else:
                                fixed_mask = np.zeros((B, T), dtype=bool)
                                fixed_mask[:, mask_start:actual_mask_end + 1] = True
                        mask_indices = torch.from_numpy(fixed_mask).to(x.device)
                else:
                    # Normal random masking
                    mask_indices = compute_mask_indices(
                        (B, T),
                        padding_mask,
                        self.mask_prob,
                        self.mask_length,
                        self.mask_selection,
                        self.mask_other,
                        min_masks=1,
                        no_overlap=self.no_mask_overlap,
                        min_space=self.mask_min_space,
                        require_same_masks=self.cfg.require_same_masks,
                        mask_dropout=self.cfg.mask_dropout,
                    )
                    mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):
        features = source

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(features)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(features)

        features = features.transpose(1, 2)

        features = self.layer_norm(features)

        orig_padding_mask = padding_mask

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        pre_encoder_features = None
        if self.cfg.ema_transformer_only:
            pre_encoder_features = features.clone()

        features = self.dropout_input(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
        else:
            x = features
            mask_indices = None

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=layer,
        )

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }

        result = {
            "losses": {},
        }

        # Check if EMA is initialized (fallback for edge cases)
        # EMA should be initialized via set_num_updates() when loading checkpoints.
        # However, if optimizer_history is missing or num_updates is not available,
        # EMA might not be initialized. This fallback handles those cases.
        if self.ema is None: 
            if self.final_proj is None:
                # Model has been converted for fine-tuning (remove_pretraining_modules called)
                # EMA is not needed in this case, but forward() should not be called with mask=True
                raise RuntimeError(
                    "EMA is None and final_proj is None. "
                    "This model has been converted for fine-tuning and cannot compute pretraining loss. "
                    "Use extract_features() with features_only=True instead."
                )
            # Fallback: Initialize EMA lazily if somehow not initialized
            # This can happen if checkpoint doesn't have optimizer_history or num_updates
            logger.warning(
                "EMA not initialized during model loading - initializing lazily from current model state. "
                "This may indicate the checkpoint is missing optimizer_history/num_updates."
            )
            self.make_ema_teacher()
            if self.cfg.ema_transformer_only:
                self.ema.model.load_state_dict(self.encoder.state_dict())
            else:
                self.ema.model.load_state_dict(self.state_dict())

        with torch.no_grad():
            self.ema.model.eval()

            # EMA model is in fp32 (ema_fp32=True), so convert inputs to float32
            # to avoid dtype mismatch errors when main model is in fp16
            if self.cfg.ema_transformer_only:
                # Convert pre_encoder_features to float32 for EMA model
                pre_encoder_features_fp32 = pre_encoder_features.float() if pre_encoder_features.dtype != torch.float32 else pre_encoder_features
                y, layer_results = self.ema.model.extract_features(
                    pre_encoder_features_fp32,
                    padding_mask=padding_mask,
                    min_layer=self.cfg.encoder_layers - self.average_top_k_layers,
                )
                y = {
                    "x": y,
                    "padding_mask": padding_mask,
                    "layer_results": layer_results,
                }
            else:
                # Convert source to float32 for EMA model
                source_fp32 = source.float() if source.dtype != torch.float32 else source
                y = self.ema.model.extract_features(
                    source=source_fp32,
                    padding_mask=orig_padding_mask,
                    mask=False,
                )

            target_layer_results = [l[2] for l in y["layer_results"]]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
                ]
                permuted = True

            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]

            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]

            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]

            if self.cfg.group_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-2:])
                    for tl in target_layer_results
                ]

            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

            y = sum(target_layer_results) / len(target_layer_results)

            if self.cfg.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.cfg.instance_norm_targets:
                y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

            if not permuted:
                y = y.transpose(0, 1)

            y = y[mask_indices]

        x = x[mask_indices]
        x = self.final_proj(x)

        sz = x.size(-1)

        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)

        result["losses"]["regression"] = loss.sum() * scale

        if "sample_size" not in result:
            result["sample_size"] = loss.numel()

        with torch.no_grad():
            result["target_var"] = self.compute_var(y)
            result["pred_var"] = self.compute_var(x.float())

        if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
            logger.error(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
            raise Exception(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
        if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
            logger.error(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )

        if self.ema is not None:
            result["ema_decay"] = self.ema.get_decay() * 1000

        return result

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(
        self, source, padding_mask, mask=False, layer=None
    ):
        res = self.forward(
            source,
            padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res

    def remove_pretraining_modules(self, last_layer=None):
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )

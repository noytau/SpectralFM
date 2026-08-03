"""
Checkpoint loading for SpectralFM evaluation.
Zero fairseq dependency — uses raw torch.load or HuggingFace transformers.
"""
import os
import re
import glob
import torch
from transformers import Data2VecAudioModel

from .model import build_model, FairseqConvFeatureExtractor


def _detect_checkpoint_type(state: dict) -> str:
    """Detect checkpoint format."""
    if "cfg" in state and "model" in state:
        return "fairseq"
    if "data2vec_audio" in state:
        return "3ae"           # 3AE recon checkpoint: embedded backbone + mirror heads
    if "encoder" in state and "layer_norm" in state and "decoder" in state:
        return "fe_recon"      # FE-only reconstruction checkpoint
    if "transformer_mirror" in state and "backbone_ckpt" in state:
        return "tr_recon"      # transformer-mirror reconstruction (backbone stored separately)
    return "state_dict"


# Aux monitoring decoders saved inside 3AE backbone states — not part of the model
_3AE_AUX_PREFIXES = ("fe_recon_decoder.", "trans_recon_decoder.")


def _install_mlp_projection(model, raw_sd: dict, prefix: str = "post_extract_proj.") -> bool:
    """
    If the fairseq-format state dict carries an MLP post_extract_proj
    (mlp_gelu: Linear→GELU→Linear, keys `post_extract_proj.0.*` / `.2.*`),
    replace the HF single-Linear `feature_projection.projection` with a matching
    nn.Sequential so the remapped keys land instead of being silently dropped.
    Dims are inferred from the tensors. Returns True if installed.
    """
    import torch.nn as nn

    w0 = raw_sd.get(prefix + "0.weight")
    w2 = raw_sd.get(prefix + "2.weight")
    if w0 is None or w2 is None:
        return False
    hidden, in_dim = w0.shape
    out_dim = w2.shape[0]
    proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
    proj.requires_grad_(False)
    model.feature_projection.projection = proj
    print(f"[CheckpointLoader] MLP projection detected — installed Sequential "
          f"({in_dim}→{hidden}→GELU→{out_dim}) at feature_projection.projection")
    return True


def _assert_projection_loaded(result) -> None:
    """MLP/linear projection keys must never be silently dropped by strict=False loads."""
    bad_missing = [k for k in result.missing_keys if k.startswith("feature_projection.projection")]
    bad_unexpected = [k for k in result.unexpected_keys if k.startswith("feature_projection.projection")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Projection keys did not load cleanly: "
            f"missing={bad_missing}, unexpected={bad_unexpected}. "
            "Checkpoint projection architecture does not match the built model."
        )


def load_3ae_backbone(state: dict, arch: str = "conv1d"):
    """
    Build the HF Data2VecAudioModel from the backbone embedded in a 3AE checkpoint
    (`data2vec_audio` key, fairseq key format). Handles the v1 single weight-normed
    positional conv (pos_conv.0.weight_g/v) by swapping in SingleLayerPosConv.
    """
    from .model import SingleLayerPosConv

    raw = state["data2vec_audio"]
    model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
    model.feature_extractor = FairseqConvFeatureExtractor(
        [(512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 5, 5)]
    )

    single_pos = "encoder.pos_conv.0.weight_v" in raw
    if single_pos:
        dim, group_in, kernel = raw["encoder.pos_conv.0.weight_v"].shape
        model.encoder.pos_conv_embed = SingleLayerPosConv(
            dim=dim, kernel=kernel, groups=dim // group_in
        )

    weights = _remap_fairseq_keys(raw)
    if single_pos:
        for suf in ("weight_g", "weight_v", "bias"):
            weights[f"encoder.pos_conv_embed.conv.{suf}"] = raw[f"encoder.pos_conv.0.{suf}"]
            weights.pop(f"encoder.pos_conv.0.{suf}", None)
    weights = {k: v for k, v in weights.items() if not k.startswith(_3AE_AUX_PREFIXES)}

    _install_mlp_projection(model, raw)
    result = model.load_state_dict(weights, strict=False)
    _assert_projection_loaded(result)
    print(f"[CheckpointLoader] 3AE embedded backbone loaded "
          f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})")
    return model


# Keys that exist only in fairseq and have no HF equivalent
_FAIRSEQ_ONLY_KEYS = {"_ema", "final_proj.weight", "final_proj.bias"}


def _remap_fairseq_keys(state_dict: dict) -> dict:
    """
    Remap fairseq data2vec-audio key names to HuggingFace Data2VecAudioModel key names.

    Key differences:
      fairseq                          →  HF
      self_attn.*                      →  attention.*
      self_attn_layer_norm.*           →  layer_norm.*
      fc1.*                            →  feed_forward.intermediate_dense.*
      fc2.*                            →  feed_forward.output_dense.*
      encoder.pos_conv.N.0.*          →  encoder.pos_conv_embed.layers.N.conv.*
      post_extract_proj.*              →  feature_projection.projection.*
      layer_norm.*  (top-level)        →  feature_projection.layer_norm.*
      mask_emb                         →  masked_spec_embed
    """
    remapped = {}
    for k, v in state_dict.items():
        if k in _FAIRSEQ_ONLY_KEYS:
            continue

        new_k = k
        new_k = new_k.replace(".self_attn.", ".attention.")
        new_k = new_k.replace(".self_attn_layer_norm.", ".layer_norm.")
        new_k = new_k.replace(".fc1.", ".feed_forward.intermediate_dense.")
        new_k = new_k.replace(".fc2.", ".feed_forward.output_dense.")
        new_k = re.sub(
            r"encoder\.pos_conv\.(\d+)\.0\.",
            r"encoder.pos_conv_embed.layers.\1.conv.",
            new_k,
        )
        new_k = new_k.replace("post_extract_proj.", "feature_projection.projection.")
        if new_k in ("layer_norm.weight", "layer_norm.bias"):
            new_k = "feature_projection." + new_k
        if new_k == "mask_emb":
            new_k = "masked_spec_embed"

        remapped[new_k] = v

    return remapped


class CheckpointLoader:
    """
    Four loading modes:
      1. from_hf       — HuggingFace pretrained (no local file needed)
      2. from_file     — Direct .pt file (state_dict or fairseq format)
      3. from_dir      — Directory: picks checkpoint_best.pt, checkpoint_last.pt, or latest
      4. load_multiple — List of paths or directory glob; returns [(label, model), ...]
    """

    @staticmethod
    def from_hf(model_name: str = "facebook/data2vec-audio-base", arch: str = "conv1d"):
        """Load HuggingFace pretrained data2vec-audio and attach custom feature extractor."""
        model = Data2VecAudioModel.from_pretrained(model_name)
        model = build_model(model, arch=arch)
        return model

    @staticmethod
    def from_file(path: str, arch: str = "conv1d", strict: bool = False):
        """
        Load from a .pt checkpoint file.
        Supports:
          - plain state_dict (saved with torch.save(model.state_dict(), ...))
          - fairseq checkpoint format (has 'cfg' and 'model' keys)
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)
        ckpt_type = _detect_checkpoint_type(state)

        model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
        model = build_model(model, arch=arch)

        if ckpt_type == "fairseq":
            cfg_model = state.get("cfg", {}).get("model", {})
            layers_cfg = cfg_model.get("conv_feature_layers")
            if isinstance(layers_cfg, str):
                layers_cfg = eval(layers_cfg)  # noqa: S307
            model.feature_extractor = FairseqConvFeatureExtractor(layers_cfg)
            _install_mlp_projection(model, state["model"])
            weights = _remap_fairseq_keys(state["model"])
            result = model.load_state_dict(weights, strict=False)
            _assert_projection_loaded(result)
            print(f"[CheckpointLoader] Loaded fairseq checkpoint: {path}")
            print(f"  Missing keys: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
        elif ckpt_type == "fe_recon":
            # Only the FE was trained; load encoder → feature_extractor, layer_norm → feature_projection.layer_norm
            layers_cfg = [(512,3,1),(512,3,1),(512,3,1),(512,3,1),(512,5,5)]
            model.feature_extractor = FairseqConvFeatureExtractor(layers_cfg)
            model.feature_extractor.load_state_dict(state["encoder"], strict=True)
            model.feature_projection.layer_norm.load_state_dict(state["layer_norm"], strict=True)
            print(f"[CheckpointLoader] Loaded FE-recon checkpoint (encoder + layer_norm): {path}")
        elif ckpt_type == "3ae":
            # Full backbone embedded in the checkpoint (fine-tuned weights) — use it directly
            model = load_3ae_backbone(state, arch=arch)
            print(f"[CheckpointLoader] Loaded 3AE checkpoint (embedded backbone): {path}")
        elif ckpt_type == "tr_recon":
            # The backbone is stored at backbone_ckpt; the transformer_mirror is a separate decoder
            backbone_path = state["backbone_ckpt"]
            if not os.path.isfile(backbone_path):
                raise FileNotFoundError(f"TR-recon backbone not found: {backbone_path}")
            print(f"[CheckpointLoader] TR-recon: loading backbone from {backbone_path}")
            return CheckpointLoader.from_file(backbone_path, arch=arch, strict=strict)
        else:
            result = model.load_state_dict(state, strict=strict)
            print(f"[CheckpointLoader] Loaded state_dict: {path}")

        return model

    @staticmethod
    def from_dir(checkpoint_dir: str, prefer: str = "best", arch: str = "conv1d"):
        """
        Load from a directory containing checkpoints.
        prefer: 'best'   → checkpoint_best.pt
                'last'   → checkpoint_last.pt
                'latest' → highest-numbered checkpoint_N.pt
        """
        if not os.path.isdir(checkpoint_dir):
            raise NotADirectoryError(f"Not a directory: {checkpoint_dir}")

        candidates = {
            "best": os.path.join(checkpoint_dir, "checkpoint_best.pt"),
            "last": os.path.join(checkpoint_dir, "checkpoint_last.pt"),
        }

        if prefer in ("best", "last") and os.path.isfile(candidates[prefer]):
            return CheckpointLoader.from_file(candidates[prefer], arch=arch)

        # Fall back to highest-numbered checkpoint
        numbered = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint[0-9]*.pt")),
            key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or "0"),
        )
        if numbered:
            print(f"[CheckpointLoader] Using latest numbered checkpoint: {numbered[-1]}")
            return CheckpointLoader.from_file(numbered[-1], arch=arch)

        # Last resort: any .pt file
        any_pt = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if any_pt:
            print(f"[CheckpointLoader] Using fallback checkpoint: {any_pt[0]}")
            return CheckpointLoader.from_file(any_pt[0], arch=arch)

        raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")

    @staticmethod
    def load_multiple(
        source,
        arch: str = "conv1d",
        pattern: str = "checkpoint*.pt",
    ) -> list:
        """
        Load multiple checkpoints for comparison evaluation.
        Loads HuggingFace base model once and reuses it across checkpoints to avoid
        redundant NFS reads of the ~700 MB model blob.

        source: str path to directory (loads all matching pattern) OR list of file paths.
        Returns: list of (label, model) tuples, sorted by label.
        """
        import copy

        if isinstance(source, (list, tuple)):
            paths = source
        elif os.path.isdir(source):
            paths = sorted(glob.glob(os.path.join(source, pattern)))
            if not paths:
                raise FileNotFoundError(f"No files matching '{pattern}' in {source}")
        else:
            raise ValueError(f"source must be a directory path or list of file paths, got: {source}")

        # Load HF base model once — reuse a deep copy per checkpoint
        print(f"[CheckpointLoader] Loading HF base model (once for all {len(paths)} checkpoints)...")
        base_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
        base_model = build_model(base_model, arch=arch)

        results = []
        for path in paths:
            label = os.path.splitext(os.path.basename(path))[0]
            print(f"[CheckpointLoader] Loading weights: {label}")

            state = torch.load(path, map_location="cpu", weights_only=False)
            model = copy.deepcopy(base_model)

            ckpt_type = _detect_checkpoint_type(state)

            if ckpt_type == "fairseq":
                cfg_model = state.get("cfg", {}).get("model", {})
                layers_cfg = cfg_model.get("conv_feature_layers")
                if isinstance(layers_cfg, str):
                    layers_cfg = eval(layers_cfg)  # noqa: S307
                model.feature_extractor = FairseqConvFeatureExtractor(layers_cfg)
                _install_mlp_projection(model, state["model"])
                weights = _remap_fairseq_keys(state["model"])
                result = model.load_state_dict(weights, strict=False)
                _assert_projection_loaded(result)
                print(f"  Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
            elif ckpt_type == "fe_recon":
                layers_cfg = [(512,3,1),(512,3,1),(512,3,1),(512,3,1),(512,5,5)]
                model.feature_extractor = FairseqConvFeatureExtractor(layers_cfg)
                model.feature_extractor.load_state_dict(state["encoder"], strict=True)
                model.feature_projection.layer_norm.load_state_dict(state["layer_norm"], strict=True)
                print(f"  FE-recon: loaded encoder + layer_norm")
            elif ckpt_type == "3ae":
                model = load_3ae_backbone(state, arch=arch)
                print("  3AE: loaded embedded backbone")
            elif ckpt_type == "tr_recon":
                backbone_path = state["backbone_ckpt"]
                print(f"  TR-recon: loading backbone from {backbone_path}")
                model = CheckpointLoader.from_file(backbone_path, arch=arch)
                label = label + "_backbone"
                path  = backbone_path  # report the resolved backbone path for mtime
            else:
                model.load_state_dict(state, strict=False)

            results.append((label, model, path))

        print(f"[CheckpointLoader] Loaded {len(results)} checkpoints.")
        return results

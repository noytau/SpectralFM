# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import socket
import subprocess

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, OrderedDict, Tuple
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import BinarizedAudioDataset, FileAudioDataset, SubsampleDataset
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.data.text_compressor import TextCompressionLevel

from . import FairseqTask, register_task


logger = logging.getLogger(__name__)


# ============================================================================
# Environment Detection and Path Translation for RunAI <-> Geoffrey
# ============================================================================
#
# RunAI cluster:
#   - GPUs: NVIDIA A5000 or A6000
#   - Path prefix: /storage/noy
#
# Geoffrey server (132.66.52.64):
#   - GPUs: 8x NVIDIA GeForce RTX 2080
#   - Path prefix: /mnt5/noy
#
# ============================================================================

# Path mappings between environments
RUNAI_PREFIX = "/storage/noy"
GEOFFREY_PREFIX = "/mnt5/noy"

# Known data directory patterns (for matching datasets: multi, 1m, all, 10k)
DATA_DIR_PATTERNS = [
    "single_channel_1m",
    "single_channel_10k", 
    "single_channel_all",
    "single_channel_multi",
    "nova_data",
]


def detect_current_environment() -> str:
    """
    Detect whether we're running on RunAI or Geoffrey server.
    
    Returns:
        "runai" if running on RunAI (NVIDIA A5000 or A6000 GPU)
        "geoffrey" if running on Geoffrey server (RTX 2080 GPUs, IP 132.66.52.64)
        "unknown" otherwise
    """
    # Method 1: Check GPU type
    # - RunAI: NVIDIA A5000 or A6000
    # - Geoffrey: 8x NVIDIA GeForce RTX 2080
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip().lower()
            # RunAI GPUs: A5000 or A6000
            if "a5000" in gpu_name or "a6000" in gpu_name:
                logger.info(f"Detected RunAI environment (GPU: {gpu_name})")
                return "runai"
            # Geoffrey GPUs: RTX 2080
            elif "2080" in gpu_name or "rtx 2080" in gpu_name:
                logger.info(f"Detected Geoffrey environment (GPU: {gpu_name})")
                return "geoffrey"
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
    
    # Method 2: Check hostname/IP for Geoffrey server
    try:
        hostname = socket.gethostname()
        # Get all IPs associated with this host
        ip_addr = socket.gethostbyname(hostname)
        
        # Geoffrey server IP
        if ip_addr.startswith("132.66.52"):
            logger.info(f"Detected Geoffrey environment (IP: {ip_addr})")
            return "geoffrey"
    except Exception as e:
        logger.debug(f"Hostname detection failed: {e}")
    
    # Method 3: Check if /mnt5/noy exists (Geoffrey) vs /storage/noy (RunAI)
    if os.path.exists("/mnt5/noy"):
        logger.info("Detected Geoffrey environment (path /mnt5/noy exists)")
        return "geoffrey"
    elif os.path.exists("/storage/noy"):
        logger.info("Detected RunAI environment (path /storage/noy exists)")
        return "runai"
    
    logger.warning("Could not detect environment, defaulting to 'unknown'")
    return "unknown"


def detect_checkpoint_environment(data_path: str) -> str:
    """
    Detect which environment a checkpoint was trained on based on its data path.
    
    Args:
        data_path: The data path from the checkpoint config
        
    Returns:
        "runai" if checkpoint was from RunAI
        "geoffrey" if checkpoint was from Geoffrey
        "unknown" otherwise
    """
    if data_path is None:
        return "unknown"
    
    if data_path.startswith(RUNAI_PREFIX) or data_path.startswith("/storage/"):
        return "runai"
    elif data_path.startswith(GEOFFREY_PREFIX) or data_path.startswith("/mnt5/"):
        return "geoffrey"
    
    return "unknown"


def translate_data_path(data_path: str, from_env: str, to_env: str) -> str:
    """
    Translate a data path from one environment to another.
    
    Handles paths like:
    - /storage/noy/fairseq/data/single_channel_1m/ <-> /mnt5/noy/fairseq/data/single_channel_1m/
    - /storage/noy/SpectralFM/fairseq/data/... <-> /mnt5/noy/SpectralFM/fairseq/data/...
    
    Args:
        data_path: Original data path
        from_env: Source environment ("runai" or "geoffrey")
        to_env: Target environment ("runai" or "geoffrey")
        
    Returns:
        Translated path for the target environment
    """
    if from_env == to_env:
        return data_path
    
    if from_env == "runai" and to_env == "geoffrey":
        # RunAI -> Geoffrey: /storage/noy -> /mnt5/noy
        translated = data_path.replace(RUNAI_PREFIX, GEOFFREY_PREFIX)
        # Also handle /storage/ without /noy
        translated = translated.replace("/storage/", "/mnt5/")
    elif from_env == "geoffrey" and to_env == "runai":
        # Geoffrey -> RunAI: /mnt5/noy -> /storage/noy
        translated = data_path.replace(GEOFFREY_PREFIX, RUNAI_PREFIX)
        translated = translated.replace("/mnt5/", "/storage/")
    else:
        logger.warning(f"Unknown environment translation: {from_env} -> {to_env}")
        return data_path
    
    logger.info(f"Translated data path: {data_path} -> {translated}")
    return translated


def maybe_translate_path_for_eval(data_path: str) -> str:
    """
    Check if we need to translate the data path for evaluation.
    
    This is called when loading a dataset for evaluation. If the checkpoint
    was trained in a different environment than the current one, translate
    the path accordingly.
    
    Args:
        data_path: The data path from the checkpoint config
        
    Returns:
        Translated path if needed, otherwise original path
    """
    current_env = detect_current_environment()
    checkpoint_env = detect_checkpoint_environment(data_path)
    
    if current_env == "unknown" or checkpoint_env == "unknown":
        logger.warning(
            f"Could not detect environments (current={current_env}, checkpoint={checkpoint_env}). "
            f"Using original path: {data_path}"
        )
        return data_path
    
    if current_env != checkpoint_env:
        logger.info(
            f"Environment mismatch detected! "
            f"Checkpoint from {checkpoint_env}, running on {current_env}. "
            f"Translating data path..."
        )
        translated_path = translate_data_path(data_path, checkpoint_env, current_env)
        
        # Verify the translated path exists
        if os.path.exists(translated_path):
            logger.info(f"Translated path exists: {translated_path}")
            return translated_path
        else:
            logger.warning(
                f"Translated path does not exist: {translated_path}. "
                f"Falling back to original: {data_path}"
            )
            return data_path
    
    return data_path


@dataclass
class AudioMaskingConfig:
    feature_encoder_spec: str = II("model.modalities.audio.feature_encoder_spec")
    mask_prob: float = II("model.modalities.audio.mask_prob")
    mask_prob_adjust: float = II("model.modalities.audio.mask_prob_adjust")
    mask_length: int = II("model.modalities.audio.mask_length")
    inverse_mask: bool = II("model.modalities.audio.inverse_mask")
    mask_dropout: float = II("model.modalities.audio.mask_dropout")
    clone_batch: int = II("model.clone_batch")
    expand_adjacent: bool = False
    non_overlapping: bool = False


@dataclass
class AudioPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    multi_corpus_keys: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated names for loading multi corpus datasets"})
    multi_corpus_sampling_weights: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated string of sampling weights corresponding to the multi_corpus_keys"})
    binarized_dataset: bool = field(
        default=False,
        metadata={
            "help": "if true, loads binarized dataset (useful for very large datasets). "
            "See examples/wav2vec/scripts/binarize_manifest.sh"
        },
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    tpu: bool = II("common.tpu")
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )

    rebuild_batches: bool = True
    precompute_mask_config: Optional[AudioMaskingConfig] = None

    post_save_script: Optional[str] = None

    subsample: float = 1
    seed: int = II("common.seed")


@register_task("audio_pretraining", dataclass=AudioPretrainingConfig)
class AudioPretrainingTask(FairseqTask):
    """ """

    cfg: AudioPretrainingConfig

    @classmethod
    def setup_task(cls, cfg: AudioPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg
        
        # Translate data path if running evaluation in different environment than training
        original_path = data_path
        data_path = maybe_translate_path_for_eval(data_path)
        if data_path != original_path:
            logger.info(f"Data path translated for cross-environment evaluation: {original_path} -> {data_path}")
        
        print(f"data_path = {data_path}, task_cfg = {task_cfg}")
        
        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )

        compute_mask = getattr(task_cfg, "precompute_mask_config", None) is not None
        mask_args = {}
        if compute_mask:
            mask_args = task_cfg.precompute_mask_config

        if getattr(task_cfg, "binarized_dataset", False):
            self.datasets[split] = BinarizedAudioDataset(
                data_path,
                split=split,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask=compute_mask,
                **mask_args,
            )
        else:
            if task_cfg.multi_corpus_keys is None:
                manifest_path = os.path.join(data_path, "{}.tsv".format(split))                

                self.datasets[split] = FileAudioDataset(
                    manifest_path=manifest_path,
                    sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                    max_sample_size=self.cfg.max_sample_size,
                    min_sample_size=self.cfg.min_sample_size,
                    pad=task_cfg.labels is not None or task_cfg.enable_padding,
                    normalize=task_cfg.normalize,
                    num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                    text_compression_level=text_compression_level,
                    compute_mask=compute_mask,
                    **mask_args,
                )
            else:
                dataset_map = OrderedDict()
                self.dataset_map = {}
                multi_corpus_keys = [k.strip() for k in task_cfg.multi_corpus_keys.split(",")]
                corpus_idx_map = {k: idx for idx, k in enumerate(multi_corpus_keys)}
                data_keys = [k.split(":") for k in split.split(",")]

                multi_corpus_sampling_weights = [float(val.strip()) for val in task_cfg.multi_corpus_sampling_weights.split(",")]
                data_weights = []

                for key, file_name in data_keys:
                    
                    k = key.strip()
                    manifest_path = os.path.join(data_path, "{}.tsv".format(file_name.strip()))                

                    # TODO: Remove duplication of code from the if block above
                    dataset_map[k] = FileAudioDataset(
                        manifest_path=manifest_path,
                        sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                        max_sample_size=self.cfg.max_sample_size,
                        min_sample_size=self.cfg.min_sample_size,
                        pad=task_cfg.labels is not None or task_cfg.enable_padding,
                        normalize=task_cfg.normalize,
                        num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                        text_compression_level=text_compression_level,
                        compute_mask=compute_mask,
                        corpus_key=corpus_idx_map[k],
                        **mask_args,
                    )

                    data_weights.append(multi_corpus_sampling_weights[corpus_idx_map[k]])

                self.dataset_map[split] = dataset_map
                
                if len(dataset_map) == 1:
                    self.datasets[split] = list(dataset_map.values())[0]
                else:
                    self.datasets[split] = MultiCorpusDataset(dataset_map, distribution=data_weights, seed=0, sort_indices=True)

        if getattr(task_cfg, "subsample", 1) < 1:
            self.datasets[split] = SubsampleDataset(
                self.datasets[split],
                task_cfg.subsample,
                shuffle=True,
                seed=task_cfg.seed,
            )

        if self.cfg.tpu and task_cfg.inferred_w2v_config.mask_channel_prob == 0.0:
            logger.info(
                "Pretraining on TPUs may suffer convergence "
                "issues when training with `mask_channel_prob` value of "
                "0. You may want to set this to a low value close to 0."
            )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(model_cfg, from_checkpoint)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            # if "w2v_args" in actualized_cfg:
            if hasattr(actualized_cfg, "w2v_args"):
                model_cfg.w2v_args = actualized_cfg.w2v_args

        return model

    def post_save(self, cp_path, num_updates):
        if self.cfg.post_save_script is not None:
            logger.info(f"launching {self.cfg.post_save_script}")
            import os.path as osp
            from fairseq.file_io import PathManager

            eval_cp_path = osp.join(
                osp.dirname(cp_path), f"checkpoint_eval_{num_updates}.pt"
            )

            print(cp_path, eval_cp_path, osp.dirname(cp_path))

            assert PathManager.copy(
                cp_path, eval_cp_path, overwrite=True
            ), f"Failed to copy {cp_path} to {eval_cp_path}"

            import subprocess
            import shlex

            subprocess.call(shlex.split(f"{self.cfg.post_save_script} {eval_cp_path}"))

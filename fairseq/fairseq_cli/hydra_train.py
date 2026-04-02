#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Server-aware path resolution
# /mnt5  = geoffry (local workstation)
# /storage = RunAI cluster
# ---------------------------------------------------------------------------
_MNT5_BASE = "/mnt5/noy/SpectralFM"
_STORAGE_BASE = "/storage/noy/SpectralFM"

def _detect_base_path() -> str:
    """Return the correct repo root for this server."""
    if os.path.isdir(_MNT5_BASE):
        return _MNT5_BASE
    if os.path.isdir(_STORAGE_BASE):
        return _STORAGE_BASE
    # Fallback: use whichever prefix the first path starts with
    return _MNT5_BASE


def _patch_cfg_paths(obj, src: str, dst: str):
    """Recursively replace *src* with *dst* in every string value of *obj*
    (dict or list produced by OmegaConf.to_container)."""
    if isinstance(obj, dict):
        return {k: _patch_cfg_paths(v, src, dst) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_patch_cfg_paths(v, src, dst) for v in obj]
    if isinstance(obj, str):
        return obj.replace(src, dst)
    return obj

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from fairseq import metrics
import fairseq.distributed.utils as distributed_utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults, hydra_init
from fairseq.dataclass.utils import omegaconf_no_object_check
from fairseq.utils import reset_logging
from fairseq_cli.train import main as pre_main

logger = logging.getLogger("fairseq_cli.hydra_train")

@hydra.main(config_path=os.path.join("..", "fairseq", "config"), config_name="config")
def hydra_main(cfg: FairseqConfig) -> float:
    _hydra_main(cfg)


def _hydra_main(cfg: FairseqConfig, **kwargs) -> float:
    add_defaults(cfg)

    if cfg.common.reset_logging:
        reset_logging()  # Hydra hijacks logging, fix that
    else:
        # check if directly called or called through hydra_main
        if HydraConfig.initialized():
            with open_dict(cfg):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                cfg.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)

    # --- Auto-fix hardcoded paths for the current server ---
    _base = _detect_base_path()
    _other = _STORAGE_BASE if _base == _MNT5_BASE else _MNT5_BASE
    if _other in OmegaConf.to_yaml(cfg):
        logger.info(
            f"[server-path] Replacing '{_other}' → '{_base}' in all config paths"
        )
        with open_dict(cfg):
            patched = _patch_cfg_paths(
                OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
                _other,
                _base,
            )
            cfg = OmegaConf.create(patched)
        OmegaConf.set_struct(cfg, True)

    # --- Initialize MLflow run (if available) ---
    if MLFLOW_AVAILABLE:
        _mlflow_base = "/mnt5/noy" if _base == _MNT5_BASE else "/storage/noy"
        mlflow.set_tracking_uri(f"file:{_mlflow_base}/code/mlruns")
        mlflow.set_experiment("SpectralFM")

        run_name = getattr(cfg.common, "wandb_run_name", None)
        if run_name is None or run_name == "":
            run_name = f"{cfg.model._name}_{cfg.task.data.split('/')[-1]}"

        mlflow.start_run(run_name=run_name)

        # Log key config parameters
        mlflow.log_params({
            "model": cfg.model._name,
            "task": cfg.task._name,
            "mask_prob": getattr(cfg.model, "mask_prob", None),
            "batch_size": cfg.dataset.batch_size,
            "learning_rate": cfg.optimization.lr[0],
        })

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, pre_main, **kwargs)
        else:
            distributed_utils.call_main(cfg, pre_main, **kwargs)
    except BaseException as e:
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! " + str(e))

    # get best val and return - useful for sweepers
    try:
        best_val = metrics.get_smoothed_value(
            "valid", cfg.checkpoint.best_checkpoint_metric
        )
    except:
        best_val = None

    if MLFLOW_AVAILABLE and mlflow.active_run():
        try:
            # Optionally log the best validation metric
            if "best_val" in locals() and best_val is not None:
                mlflow.log_metric(cfg.checkpoint.best_checkpoint_metric, best_val)
            mlflow.end_run()
            logger.info("MLflow run closed successfully.")
        except Exception as e:
            logger.warning(f"Failed to close MLflow run cleanly: {e}")

    if best_val is None:
        best_val = float("inf")

    return best_val


def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    hydra_init(cfg_name)
    hydra_main()


if __name__ == "__main__":
    cli_main()
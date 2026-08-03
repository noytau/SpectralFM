# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

logger = logging.getLogger(__name__)

__all__ = []

# Vision-only tasks, unused by SpectralFM's audio pretraining; their imports
# are fragile (rely on an ambient `data2vec` top-level import) so a failure
# here must not take down audio task/model registration.
try:
    from .image_pretraining import ImagePretrainingTask, ImagePretrainingConfig
    from .image_classification import ImageClassificationTask, ImageClassificationConfig
    from .mae_image_pretraining import MaeImagePretrainingTask, MaeImagePretrainingConfig

    __all__ += [
        "ImageClassificationTask",
        "ImageClassificationConfig",
        "ImagePretrainingTask",
        "ImagePretrainingConfig",
        "MaeImagePretrainingTask",
        "MaeImagePretrainingConfig",
    ]
except ImportError as e:
    logger.warning(f"Skipping data2vec vision tasks (not used by SpectralFM): {e}")
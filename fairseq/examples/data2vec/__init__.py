import logging

logger = logging.getLogger(__name__)

from . import models, tasks

try:
    from . import criterions
except ImportError as e:
    logger.warning(f"Skipping data2vec.criterions (not present): {e}")

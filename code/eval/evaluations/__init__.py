from . import embedding_similarity as _emb
from . import signal_reconstruction as _recon
from . import noise_robustness as _noise
from . import checkpoint_comparison as _cmp
from . import clustering as _clust
from . import label_regression as _labelreg


class EmbeddingSimilarityEval:
    run = staticmethod(_emb.run)


class SignalReconstructionEval:
    run = staticmethod(_recon.run)


class NoiseRobustnessEval:
    run = staticmethod(_noise.run)


class CheckpointComparisonEval:
    run = staticmethod(_cmp.run)


class ClusteringEval:
    run = staticmethod(_clust.run)


class LabelRegressionEval:
    run = staticmethod(_labelreg.run)

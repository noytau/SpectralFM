from fairseq.tasks import FairseqTask, register_task
from fairseq.data.pkl_feature_dataset import PklFeatureDataset

@register_task("pkl_pretraining")
class PklPretrainingTask(FairseqTask):
    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls(cfg)

    def load_dataset(self, split, **kwargs):
        manifest = os.path.join(self.cfg.data, f"{split}.tsv")
        self.datasets[split] = PklFeatureDataset(manifest)

from run_experiment import ExperimentRunner


class EvalExperiment(ExperimentRunner):
    def __init__(self, model, config, eval_method):
        super().__init__(...)
        self.model = model
        self.eval_method = eval_method
        self.eval_mode = True  # skip training

    def run_evaluation(self, batch):
        pass
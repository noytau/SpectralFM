from run_experiment import ExperimentRunner
from testing import Testing
from model_loader import *
from compute_stats import *

class EvalExperiment(ExperimentRunner):
    def __init__(self, eval_args, model=None):
        super().__init__(...)
        self.args = eval_args
        self.eval_method = eval_args.eval_method
        self.eval_mode = True  # skip training
        self.model = model
        print(f"Initialized evaluation with method: {self.eval_method}, test method: {self.args.test_method}, stack method: {self.args.stack_method}")

    def evaluate_signal_completion(self, test_data):
        print(f"Evaluating signal completion on {len(test_data)} samples...")
        pass

    def evaluate_noise_robustness(self, noise_levels, test_data):
        pass

    def compare_embeddings(self,test_data):
        pass

    def classifier_head(self, test_data):
        pass

    def run_evaluation(self):
        df = self.prepare_data() # parse data to df

        # Get evaluation data
        test = Testing(model=None, df=df, test_method=self.args.test_method, stack_method=self.args.stack_method)  # fixme replace for args
        test_data = test.get_test_data()

        # Run evaluation based on method on test_data
        if self.eval_method == "signal_completion":
            self.evaluate_signal_completion(test_data)
        elif self.eval_method == "noise_robustness":
            self.evaluate_noise_robustness(test_data, noise_levels=[0.1, 0.2, 0.3])
        else:
            print(f"Unknown evaluation method: {self.eval_method}")


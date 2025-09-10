import pandas as pd
from data_parser import run_data_parser
from model_loader import compute_stack_from_input_spectograms
from compute_stats import Stats

class Testing:
    def __init__(self, model, stack_dataset):
        """
        model: your trained model object
        dataset: a dataset object or DataFrame
        stack_column: name of the column that denotes the stack
        input_column: the feature column (or key)
        """
        self.model = model
        self.stack_dataset = stack_dataset

    def run_all_tests(self):
        print("Running all test types...")
        self.test_out_of_distribution()
        self.test_in_distribution_stack_holdout()
        self.test_in_distribution_partial_stack()

    def test_out_of_distribution(self, ood_samples=None):
        """
        Evaluate the model on fully unseen samples.
        ood_samples: list of unseen samples (or provide filtering logic)
        """
        print("Testing Out-of-Distribution...")
        # run on /mnt5/noy/nova_samples/multi_chnl

    def test_in_distribution_stack_holdout(self, input_stack, heldout_stacks=None): # fixme input_stack
        """
        Leave out entire stacks for testing (model never saw these stacks during training)
        heldout_stacks : list of stacks to hold out from training
        """
        print("Testing In-Distribution (held-out stacks)...")
        if heldout_stacks is None:
            # Automatically select a few stacks to hold out
            unique_stacks = self.dataset[self.stack_column].unique()
            heldout_stacks = unique_stacks[:2]  # First 2 as example

        test_data = {key: input_stack[key] for key in input_stack if key not in heldout_stacks}
        return test_data

    def test_in_distribution_partial_stack(self, input_stack, holdout_ratio=0.3):
        """
        For each stack, hold out a portion (e.g., 30%) of samples
        """
        print("Testing In-Distribution (partial stack holdout)...")
        test_data = []

        for stack_name, group in input_stack.items():
            n = len(group)
            n_holdout = int(n * holdout_ratio)
            test_data.append(group.sample(n=n_holdout))

        test_data = pd.concat(test_data)
        return test_data


    def _evaluate(self, predictions, ground_truth_data):
        """
        Evaluate predictions vs ground truth.
        """
        # Dummy example
        print(f"Evaluating {len(predictions)} predictions...")
        # You can add MSE, accuracy, etc. here


    def evaluate_signal_completion(self, mask_experiment):
        """
        Test model's ability to reconstruct masked signals.
        mask_ratios: list of mask ratios to test
        """
        print("Evaluating Signal Completion...")


    def evaluate_noise_robustness(self, noise_levels):
        """
        Test model robustness to varying noise levels.
        noise_levels: list of noise levels to test
        """
        print("Evaluating Noise Robustness...")
        for level in noise_levels:
            print(f"Testing with noise level: {level}")
            # Add noise to test data and evaluate

if __name__ == "__main__":
    get_stacks_by_index = False # fixme get from args or config
    # get sample data
    samples_path = '/mnt5/noy/nova_samples/debug_chnl'  # fixme get from args or config
    df = run_data_parser(samples_path)  # returns df
    if get_stacks_by_index:
        input_stack = compute_stack_from_input_spectograms(df)
    else:
        # compute stack from k-means similarities
        stats = Stats(df=df, argparse=None)
        input_stack = stats.cluster_vectors(df, 12, visualize=False) # fixme get best k
    test = Testing(model=None, stack_dataset=input_stack)  # fixme replace None

    test.test_in_distribution_stack_holdout(input_stack, [1,5,10,9])  # example stacks
    # test.test_in_distribution_partial_stack(input_stack, holdout_ratio=0.3)

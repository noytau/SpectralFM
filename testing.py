import pandas as pd
from data_parser import run_data_parser
from model_loader import compute_stack_from_input_spectograms
from compute_stats import Stats

class Testing:
    def __init__(self, df, test_method, stack_method, model=None):
        """
        model: your trained model object
        dataset: a dataset object or DataFrame
        stack_column: name of the column that denotes the stack
        input_column: the feature column (or key)
        """
        self.df = df
        self.test_method = test_method if test_method in ["index_out_of_distribution", "index_in_distribution_stack_holdout", "test_in_distribution_partial_stack"] else "index_in_distribution_stack_holdout"
        self.stack_method =  stack_method if stack_method in ["kmeans", "index"] else "index"

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

    def test_in_distribution_stack_holdout(self, input_stack, heldout_stacks=None):
        """
        Leave out entire stacks for testing (model never saw these stacks during training)
        heldout_stacks : list of stacks to hold out from training
        """
        print("Testing In-Distribution (held-out stacks)...")
        if heldout_stacks is None:
            # Automatically select a few stacks to hold out
            unique_stacks = list(input_stack.keys())
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

    def get_test_data(self):
        # get stacks for data division
        division_type = self.stack_method
        input_stack = self.divide_data_into_stacks(self.df)

        if self.test_method == "index_out_of_distribution":
            return self.test_out_of_distribution(input_stack)
        elif self.test_method == "index_in_distribution_stack_holdout":
            # return self.test_in_distribution_stack_holdout(input_stack)
            return self.test_in_distribution_stack_holdout(input_stack, heldout_stacks=range(0,50)) # fixme
        elif self.test_method == "test_in_distribution_partial_stack":
            return self.test_in_distribution_partial_stack(input_stack)
        return self.test_out_of_distribution(input_stack)

    def divide_data_into_stacks(self, df):
        """
        Divide the DataFrame into stacks based on some criteria.
        Here we use k-means clustering as an example.
        """
        if self.stack_method == "index":
            input_stack = compute_stack_from_input_spectograms(df)
        else:
            # compute stack from k-means similarities
            stats = Stats(df=df, argparse=None)
            input_stack = stats.cluster_vectors(df, 12, visualize=False) # fixme get best k?
        return input_stack



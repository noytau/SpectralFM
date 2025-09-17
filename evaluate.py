from run_experiment import ExperimentRunner
from testing import Testing
from model_loader import *
from compute_stats import *
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class EvalExperiment(ExperimentRunner):
    def __init__(self, eval_args):
        super().__init__(...)
        self.args = eval_args
        self.eval_method = eval_args.eval_method
        self.model = eval_args.model
        self.eval_mode = True  # skip training
        print(f"Initialized evaluation with method: {self.eval_method}, test method: {self.args.test_method}, stack method: {self.args.stack_method}")

    def evaluate_signal_completion(self, test_data):
        print(f"Evaluating signal completion on {len(test_data)} samples...")
        total_loss = 0
        all_rows = []

        for stack_idx, samples in test_data.items():

            dataloader, masked_data = prepare_masked_dataloader(samples, args=self.args)
            clean_inputs = torch.stack([item["data"] for item in masked_data]).to(self.model.device)
            masked_inputs = torch.stack([item["masked_data"] for item in masked_data])
            outputs, embeddings = evaluate_embedding_from_model(self.args, dataloader)
            predicted_signal = self.model.completion_head(embeddings)

            # Add predictions to masked_data
            for i, item in enumerate(masked_data):
                item["predicted"] = predicted_signal[i].detach().cpu()
                item["embedding"] = embeddings[i].detach().cpu()

                all_rows.append({
                    "inputs": item["data"].numpy(),
                    "masked": item["masked_data"].numpy(),
                    "predicted": item["predicted"].numpy(),
                    "embedding": item["embedding"].numpy(),
                    "stack_idx": stack_idx,
                    "mse_less": F.mse_loss(item["predicted"], item["data"]).item()
                })

            loss = F.mse_loss(predicted_signal, clean_inputs)
            print("MSE Loss for stack {}: {:.6f}".format(stack_idx, loss.item()))
            total_loss += loss.item()

        avg_loss = total_loss / len(test_data)
        print("Average MSE Loss across all stacks: {:.6f}".format(avg_loss))

        df = pd.DataFrame(all_rows)
        return df

    def add_noisy_embeddings_to_df(self, df, model, device='cpu', noise_std=0.05):
        model.eval()
        model.to(device)

        noisy_embeddings = []
        for i in range(len(df)):
            x = torch.tensor(df.iloc[i]['inputs'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            noise = torch.randn_like(x) * noise_std
            x_noisy = x + noise
            with torch.no_grad():
                emb = model(x_noisy).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            noisy_embeddings.append(emb)

        df["noisy_embedding"] = noisy_embeddings
        return df

    def evaluate_embedding_robustness_to_noise_from_df(self, df, max_samples=None):
        """
        Compare clean vs noisy embeddings stored in df using cosine similarity.

        Assumes:
            - df["embedding"]: clean embeddings
            - df["noisy_embedding"]: noisy embeddings

        Returns:
            List of cosine similarities
        """
        from torch.nn.functional import cosine_similarity

        similarities = []
        sample_indices = range(len(df)) if max_samples is None else range(min(max_samples, len(df)))

        for i in sample_indices:
            clean_emb = torch.tensor(df.iloc[i]["embedding"]).float()
            noisy_emb = torch.tensor(df.iloc[i]["noisy_embedding"]).float()
            sim = cosine_similarity(clean_emb.unsqueeze(0), noisy_emb.unsqueeze(0)).item()
            similarities.append(sim)

        return similarities

    def compare_embeddings(self,test_data):
        pass

    def classifier_head(self, test_data):
        pass

    def print_eval_summary(self, eval_method, test_data, results):
        print(f"Evaluation Summary for method: {eval_method}")
        print(f"Number of test samples: {len(test_data)}")
        print(f"Results: {results}")


    def run_evaluation(self):
        df = self.prepare_data() # parse data to df

        # Get evaluation data
        test = Testing(model=None, df=df, test_method=self.args.test_method, stack_method=self.args.stack_method)  # fixme replace for args
        test_data = test.get_test_data()

        # Run evaluation based on method on test_data
        if self.eval_method == "signal_completion":
            signal_comp_df = self.evaluate_signal_completion(test_data)


            stats = Stats(df=signal_comp_df, argparse=self.args)
            stats.output_dir = "/mnt5/noy/code/logs/eval_outputs/16-09-25/mask=0.25-epoch=1_batch=32_loss_fn=mse" # fixme pass from args
            #stats.plot_best_and_worst_predictions(signal_comp_df, k=3)

            #result = stats.retrieve_k_similar_inputs_by_embedding(signal_comp_df, query_index=7, k=3)

            #stats.visualize_embedding_similarity_in_input_space(signal_comp_df, query_index=24, k=5)
            #stats.visualize_input_similarity_in_input_space(df=signal_comp_df, query_index=24, k=5)

            #results = stats.compare_embedding_vs_input_similarity(df=signal_comp_df, k=5, max_samples=500)
            #stats.plot_similarity_comparison(results)
            # print("Similar sample indices:", result["indices"])

            # fixme move to if noise robustness?
            self.add_noisy_embeddings_to_df(signal_comp_df, model=self.model, device=self.model.device, noise_std=0.1)
            sims = self.evaluate_embedding_robustness_to_noise_from_df(signal_comp_df)
            stats.plot_noise_robustness_histogram(sims, noise_std=0.3)
            stats.plot_robustness_scatter(signal_comp_df, sims, noise_std=0.3)




        elif self.eval_method == "noise_robustness":
            pass
            # self.evaluate_embedding_robustness_to_noise(test_data, noise_levels=[0.1, 0.2, 0.3])
        else:
            print(f"Unknown evaluation method: {self.eval_method}")


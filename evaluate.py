import datetime

from run_experiment import ExperimentRunner
from testing import Testing
from model_loader import *
from compute_stats import *
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import os


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

    def add_noisy_embeddings_to_df(self, df, noisy_data_columns, model, device='cpu'):

        model.eval()
        model.to(device)

        # Find all columns starting with "data_"

        for col in noisy_data_columns:
            embeddings_list = []
            for i in range(len(df)):
                x_noisy = torch.tensor(df.iloc[i][col], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(x_noisy).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings_list.append(emb)
            df["embedding_" + col] = embeddings_list

        return df

    def evaluate_embedding_robustness_to_noise_from_df(self, df, noise_cols, max_samples=None):
        """
        Compare clean vs noisy embeddings and data for each noise type.

        Args:
            df (pd.DataFrame): dataframe with columns ["data", "embeddings", "data_<noise>", "embedding_<noise>"]
            noise_cols (list): list of noise column names (e.g. ["data_gaussian_0_001", "data_shot_low", ...])
            max_samples (int, optional): limit number of samples to compute.
        """
        sample_indices = range(len(df)) if max_samples is None else range(min(max_samples, len(df)))

        for noise_col in noise_cols:
            emb_col = "embedding_" + noise_col
            data_sims = []
            emb_sims = []

            for i in sample_indices:
                clean_data = torch.tensor(df.iloc[i]["data"]).float()
                noisy_data = torch.tensor(df.iloc[i][noise_col]).float()
                clean_emb = torch.tensor(df.iloc[i]["embeddings"]).float()
                noisy_emb = torch.tensor(df.iloc[i][emb_col]).float()

                # Compute cosine similarities
                data_sim = cosine_similarity(clean_data.unsqueeze(0), noisy_data.unsqueeze(0)).item()
                emb_sim = cosine_similarity(clean_emb.unsqueeze(0), noisy_emb.unsqueeze(0)).item()

                data_sims.append(data_sim)
                emb_sims.append(emb_sim)

            # Save results as new columns
            df[f"data_noise_similarity_{noise_col.replace('data_', '')}"] = data_sims
            df[f"embedding_noise_similarity_{noise_col.replace('data_', '')}"] = emb_sims

        return df


    def print_eval_summary(self, eval_method, test_data, results):
        print(f"Evaluation Summary for method: {eval_method}")
        print(f"Number of test samples: {len(test_data)}")
        print(f"Results: {results}")

    def run_evaluation(self):
        df = self.prepare_data()  # parse data to df

        # Get data for evaluation based on test method
        test = Testing(model=None, df=df, test_method=self.args.test_method, stack_method=self.args.stack_method)
        test_data = test.get_test_data()

        all_data, noise_cols = self.create_comp_df(test_data)

        comparison_results = self.compare_similarity_by_stack_membership(all_data, k=5)
        print(comparison_results.head())

        self.evaluate_stats_and_visualizations(all_data, comparison_results, noise_cols)


    def create_comp_df(self, test_data):
        # Create dataframe from test_data and preserve original indices
        all_data = pd.DataFrame(
            columns=['data', 'masked_data', 'embeddings', 'stack_idx', 'noisy_data', 'noisy_embeddings'])

        # Keep the original index from the full dataset
        all_data['index'] = test_data.index.values
        all_data['data'] = test_data.loc[:, test_data.columns != 'stack_idx'].values.tolist()
        all_data['stack_idx'] = test_data['stack_idx'].values

        # fixme: understand if I need to evaluate embedding like so, or use no_grad (like in noisy)
        # Run dataloader + model
        dataloader, all_data = prepare_masked_dataloader(all_data, args=self.args)
        outputs, embeddings = evaluate_embedding_from_model(self.args, dataloader)
        all_data['embeddings'] = [emb.detach().cpu() for emb in embeddings]

        all_data, noisy_cols = self.add_noisy_data_to_df(all_data)
        all_data = self.add_noisy_embeddings_to_df(all_data, noisy_cols, self.model, device=self.model.device)

        return all_data, noisy_cols

    # fixme noy: ask nimrod about the noise additions, specifically shot noise gaussian
    def add_noisy_data_to_df(self, df):
        noise_data_cols = {
            "data_gaussian_std": [],
            "data_gaussian_mean": [],
            "data_shot_low": [],
            "data_shot_high": [],
            "data_gain_low": [],
            "data_gain_high": []
        }
        # fixme chack values of data (-1 to 1?)
        for x in df["data"]:
            x = np.array(x)
            noise_data_cols["data_gaussian_std"].append(x + np.random.normal(0, 0.01, size=len(x)))
            noise_data_cols["data_gaussian_mean"].append(x + np.random.normal(2, 0.001, size=len(x)))
            noise_data_cols["data_shot_low"].append(np.random.poisson((x - x.min() + 1e-4) * 0.1) / 0.1 - 1e-4)
            noise_data_cols["data_shot_high"].append(np.random.poisson((x - x.min() + 1e-4) * 0.05) / 0.05 - 1e-4)
            noise_data_cols["data_gain_low"].append(x * np.random.normal(1, 0.05))
            noise_data_cols["data_gain_high"].append(x * np.random.normal(1, 0.1))

        for name, values in noise_data_cols.items():
            df[name] = values
        return df, noise_data_cols

    def evaluate_stats_and_visualizations(self, signal_comp_df, match_df, noise_cols, k=5):

        #output_dir = f"{self.args.output_path}/eval_outputs/{datetime.today().strftime('%d-%m-%y')}_{self.args.model_name}" # fixme pass from args
        stats = Stats(df=signal_comp_df, argparse=self.args, eval=True)
        # Compute diff in match length between embedding and input space
        match_df["match_diff"] = match_df.apply(
            lambda row: len(set(row["embedding_stack_matches"])) - len(set(row["input_stack_matches"])),
            axis=1
        )

        # Normalize match_diff to a 0–100 score
        match_df["match_score"] = ((match_df["match_diff"] + 5) / 10) * 100
        avg_score = match_df["match_score"].mean()
        score_row = pd.DataFrame([{
            "index": "average_score",
            "match_score": avg_score
        }])

        # Save match_df to CSV for inspection
        match_df_with_score = pd.concat([match_df, score_row], ignore_index=True)
        match_df_with_score.to_csv(os.path.join(stats.output_dir, "match_df.csv"), index=False)

        # Get k best (most agreement) and k worst (most disagreement)
        best_matches = match_df.nsmallest(k, "match_diff")
        worst_matches = match_df.nlargest(k, "match_diff")

        print(f"\nTop-{k} best matches (agreement):", best_matches["index"].tolist())
        print(f"Top-{k} worst matches (disagreement):", worst_matches["index"].tolist())

        for idx in best_matches["index"]:
            stats.compare_embedding_vs_input_similarity(
                df=signal_comp_df,
                match_df=match_df,
                query_index=idx,
                k=5
            )

        for idx in worst_matches["index"]:
            stats.compare_embedding_vs_input_similarity(
                df=signal_comp_df,
                match_df=match_df,
                query_index=idx,
                k=5
            )


        # fixme noy not sure its necessary to iterate over all best/worst - maybe calculate avg similarity for best/worst sets?
        signal_comp_df = self.evaluate_embedding_robustness_to_noise_from_df(signal_comp_df, noise_cols)
        # Iterate over each noise type and compute best/worst indices for each type independently
        for noise_col in noise_cols:
            emb_sim_col = f"embedding_noise_similarity_{noise_col.replace('data_', '')}"
            if emb_sim_col not in signal_comp_df.columns:
                print(f"Skipping {noise_col}, missing column {emb_sim_col}")
                continue

            noise_sim = np.array(signal_comp_df[emb_sim_col])
            best_noisy_idx = noise_sim.argsort()[-k:][::-1]
            worst_noisy_idx = noise_sim.argsort()[:k]

            print(f"\nNoise type: {noise_col}")
            print(f"Top-{k} best noisy indices ({noise_col}):", best_noisy_idx.tolist())
            print(f"Top-{k} worst noisy indices ({noise_col}):", worst_noisy_idx.tolist())

            stats.plot_noise_robustness_histogram(signal_comp_df, [noise_col])

            for idx in best_noisy_idx:
                stats.plot_noisy_vs_clean_spectrogram(signal_comp_df, idx, [noise_col], status="Best")

            for idx in worst_noisy_idx:
                stats.plot_noisy_vs_clean_spectrogram(signal_comp_df, idx, [noise_col], status="Worst")

    def compare_similarity_by_stack_membership(
            self,
            df,
            k=5,
            include_seen_in_stack_row=True,
            fill_with_seen_if_not_enough=True
    ):
        """
        Compute similarities for each row.
        - include_seen_in_stack_row=False: exclude query and top-k already seen from same-stack list
        - fill_with_seen_if_not_enough=True: if not enough unseen same-stack, fill with seen ones
        """
        from sklearn.metrics.pairwise import cosine_similarity

        input_matrix = np.stack(df['data'].to_numpy())
        embedding_matrix = np.stack(df['embeddings'].to_numpy())

        results = []

        for idx in range(len(df)):
            query_stack = df.iloc[idx]['stack_idx']
            query_input = input_matrix[idx].reshape(1, -1)
            query_embedding = embedding_matrix[idx].reshape(1, -1)

            input_sims = cosine_similarity(query_input, input_matrix)[0]
            embedding_sims = cosine_similarity(query_embedding, embedding_matrix)[0]

            topk_input_idx = input_sims.argsort()[::-1][1:k + 1]
            topk_embedding_idx = embedding_sims.argsort()[::-1][1:k + 1]

            input_stack_matches = [int(i) for i in topk_input_idx if df.iloc[i]['stack_idx'] == query_stack]
            emb_stack_matches = [int(i) for i in topk_embedding_idx if df.iloc[i]['stack_idx'] == query_stack]

            # Same-stack samples
            same_stack_df = df[df["stack_idx"] == query_stack].copy()

            # Always drop the query index
            same_stack_df = same_stack_df.drop(index=idx, errors='ignore')

            if not include_seen_in_stack_row:
                seen = set(topk_input_idx) | set(topk_embedding_idx)
                same_stack_df = same_stack_df[~same_stack_df.index.isin(seen)]

                if len(same_stack_df) < k and fill_with_seen_if_not_enough:
                    # Re-include seen (but NOT query)
                    refill_df = df[df["stack_idx"] == query_stack].copy()
                    refill_df = refill_df.drop(index=idx, errors='ignore')
                    refill_df["cosine_sim"] = refill_df.index.map(lambda i: input_sims[i])
                    same_stack_df = refill_df.sort_values("cosine_sim", ascending=False)
            else:
                # Still need to drop query index
                same_stack_df["cosine_sim"] = same_stack_df.index.map(lambda i: input_sims[i])
                same_stack_df = same_stack_df.sort_values("cosine_sim", ascending=False)

            same_stack_sorted = same_stack_df.head(k)
            same_stack_idxs = same_stack_sorted.index.tolist()
            same_stack_sims = same_stack_sorted["cosine_sim"].tolist()
            results.append({
                    "index": idx,
                    "stack_idx": query_stack,
                    "embedding_neighbors": topk_embedding_idx.tolist(),
                    "embedding_similarities": embedding_sims[topk_embedding_idx].tolist(),
                    "embedding_stack_matches": emb_stack_matches,
                    "input_neighbors": topk_input_idx.tolist(),
                    "input_similarities": input_sims[topk_input_idx].tolist(),
                    "input_stack_matches": input_stack_matches,
                    "same_stack_neighbors": same_stack_idxs,
                    "same_stack_similarities": same_stack_sims
                })

        return pd.DataFrame(results)

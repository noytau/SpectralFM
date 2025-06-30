from transformers import Trainer
import torch

class SelfSupervisedDataCollator:
    def __init__(self, feature_extractor, device):
        self.feature_extractor = feature_extractor
        self.device = device

    def __call__(self, batch):
        masked = [item["masked_data"] for item in batch]
        original = [item["data"] for item in batch]

        masked_inputs = self.feature_extractor(masked, sampling_rate=16000, return_tensors="pt", padding=True)
        original_inputs = self.feature_extractor(original, sampling_rate=16000, return_tensors="pt", padding=True)

        # return {
        #     "masked_inputs": {k: v.to(self.device) for k, v in masked_inputs.items()},
        #     "original_inputs": {k: v.to(self.device) for k, v in original_inputs.items()}
        # }
        return {
            "masked_inputs": masked_inputs,
            "original_inputs": original_inputs
        }


class SelfSupervisedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # fixme debug
        print("Masked input shape:", inputs["masked_inputs"]["input_values"].shape)

        masked_out = model(**inputs["masked_inputs"]).last_hidden_state
        original_out = model(**inputs["original_inputs"]).last_hidden_state

        min_len = min(masked_out.shape[1], original_out.shape[1])
        masked_out = masked_out[:, :min_len, :]
        original_out = original_out[:, :min_len, :]

        loss = torch.nn.functional.mse_loss(masked_out, original_out)
        return (loss, masked_out) if return_outputs else loss
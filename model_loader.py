from transformers import AutoFeatureExtractor, Data2VecVisionModel
import torch


def load_model_and_extractor(model_name: str):
    """
    Load the Data2VecVision model and feature extractor.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the model and the feature extractor.
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = Data2VecVisionModel.from_pretrained(model_name)
    # Set the model to evaluation mode
    model.eval()
    return model, feature_extractor

def mask_spectogram(spectrogram: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
    """
    Apply masking to the spectrogram.

    Args:
        spectrogram (torch.Tensor): The input spectrogram tensor.
        mask_ratio (float): The ratio of the spectrogram to be masked.

    Returns:
        torch.Tensor: The masked spectrogram tensor.
    """
    # Get the dimensions of the spectrogram
    batch_size, channels, height, width = spectrogram.shape

    # Calculate the number of pixels to mask
    num_masked_pixels = int(height * width * mask_ratio)

    # Generate random indices for masking
    mask_indices = torch.randperm(height * width)[:num_masked_pixels]

    # Create a mask tensor
    mask = torch.ones_like(spectrogram)
    for idx in mask_indices:
        h = idx // width
        w = idx % width
        mask[:, :, h, w] = 0  # Set the pixel to zero

    return spectrogram * mask  # Apply the mask to the spectrogram

def load_and_mask_spectrogram(model_name: str, spectrogram: torch.Tensor, mask_ratio: float = 0.5):
    """
    Load the model and feature extractor, then apply masking to the spectrogram.

    Args:
        model_name (str): The name of the model to load.
        spectrogram (torch.Tensor): The input spectrogram tensor.
        mask_ratio (float): The ratio of the spectrogram to be masked.

    Returns:
        tuple: A tuple containing the model, feature extractor, and masked spectrogram.
    """
    model, feature_extractor = load_model_and_extractor(model_name)
    masked_spectrogram = mask_spectogram(spectrogram, mask_ratio)
    return model, feature_extractor, masked_spectrogram

def apply_model_to_spectrogram(model, feature_extractor, spectrogram: torch.Tensor):
    """
    Apply the model to the masked spectrogram.

    Args:
        model: The loaded model.
        feature_extractor: The feature extractor.
        spectrogram (torch.Tensor): The masked spectrogram tensor.

    Returns:
        torch.Tensor: The output from the model.
    """
    # Preprocess the spectrogram using the feature extractor
    inputs = feature_extractor(spectrogram, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state  # Return the last hidden state as output

if __name__ == "__main__":
    # Example usage
    model_name = "facebook/data2vec-vision-base"
    # Assuming spectrogram is a torch.Tensor of shape (batch_size, channels, height, width)
    spectrogram = torch.randn(1, 1, 1, 245)  # Example random spectrogram

    model, feature_extractor, masked_spectrogram = load_and_mask_spectrogram(model_name, spectrogram)
    output = apply_model_to_spectrogram(model, feature_extractor, masked_spectrogram)

    print("Masked Spectrogram Shape:", masked_spectrogram.shape)
    print("Model Output Shape:", output.shape)

    print("Model and feature extractor loaded successfully.")




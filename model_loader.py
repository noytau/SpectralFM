from transformers import AutoFeatureExtractor, Data2VecVisionModel, Data2VecAudioForAudioFrameClassification
from datasets import load_dataset
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

def load_and_apply_audio_model():
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation",
                           trust_remote_code=True)
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/data2vec-audio-base-960h")
    model = Data2VecAudioForAudioFrameClassification.from_pretrained("facebook/data2vec-audio-base-960h")

    # audio file is decoded on the fly
    inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.sigmoid(logits[0])
    # labels is a one-hot array of shape (num_frames, num_speakers)
    labels = (probabilities > 0.5).long()
    labels[0].tolist()





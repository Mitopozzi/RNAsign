#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Initializes and loads a pre-trained PyTorch model from a state dictionary.

    Args:
        model_class (Type[nn.Module]): The class of the model to initialize (e.g., EnhancedGraphSAGE).
        in_channels (int): The number of input channels for the model.
        hidden_channels (int): The number of hidden channels for the model.
        model_path (str): The file path to the saved model state dictionary (.pth file).
        device (torch.device): The device (CPU or CUDA) to load the model onto.

    Returns:
        nn.Module: The loaded model, moved to the specified device and set to evaluation mode.
        
    Raises:
        FileNotFoundError: If the model file does not exist at the given path.
"""

import torch
import torch.nn as nn
from typing import Type
import joblib
from typing import Any
import sys

def load_prediction_model(
    model_class: Type[nn.Module],
    in_channels: int,
    hidden_channels: int,
    model_path: str,
    device: torch.device
) -> nn.Module:

    print("\n--- LOADING MODEL FOR EMBEDDING PREDICTION ---")
    try:
        # 1. Initialize the model architecture
        model = model_class(in_channels=in_channels, hidden_channels=hidden_channels)
        
        # 2. Load the saved weights from the file
        print(f"\nLoading model weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        # 3. Load the weights into the model structure
        model.load_state_dict(state_dict)
        
        # 4. Move the model to the correct device (CPU or GPU)
        model.to(device)
        
        # 5. Set the model to evaluation mode
        # This disables layers like Dropout for consistent predictions.
        model.eval()
        
        print(f"\n✓ Model successfully loaded and set to evaluation mode on {device}.")
        return model

    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at '{model_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred during model loading: {e}")
        sys.exit(1)
        

def load_classifier(model_path: str) -> Any:
    """
    Loads a pre-trained scikit-learn model from a file.

    Args:
        model_path (str): The file path to the saved model (.joblib file).

    Returns:
        Any: The loaded scikit-learn model object, ready for prediction.
        
    Raises:
        FileNotFoundError: If the model file does not exist at the given path.
    """
    print("\n--- LOAD CLASSIFIER ---")
    try:
        print(f"Loading model from: {model_path}")
        
        # Load the entire model object from the file
        model = joblib.load(model_path)
        
        print("✓ Model successfully loaded.")
        return model

    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at '{model_path}'.")
        raise
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred during model loading: {e}")
        raise
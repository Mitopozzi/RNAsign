#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Fits a StandardScaler on an entire DataFrame, transforms the features,
    and optionally saves the scaler object, with a progress bar.

    Args:
        df_features (pd.DataFrame): The DataFrame containing features to scale.
        features_to_scale (list): A list of column names to be scaled.
        save_path (str | None, optional): File path to save the fitted scaler. 
                                         If None, the scaler is not saved. Defaults to None.

    Returns:
        tuple[pd.DataFrame, StandardScaler]: A tuple containing:
            - The DataFrame with the specified features scaled.
            - The scaler object that was fitted on the data.
"""
from sklearn.preprocessing import StandardScaler
from typing import Any
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os
from tqdm import tqdm

def scale_features(
    df_features: pd.DataFrame, 
    features_to_scale: list, 
    save_path: str | None = None
) -> tuple[pd.DataFrame, StandardScaler]:

    df_scaled = df_features.copy()
    scaler = StandardScaler()
    
    # Wrap the sequence of operations in a tqdm context manager
    with tqdm(total=3, desc="Scaling Features") as pbar:
        # Step 1: Fit the StandardScaler
        pbar.set_description("Fitting scaler")
        scaler.fit(df_scaled[features_to_scale])
        pbar.update(1)
        
        # Step 2: Save the fitted scaler
        pbar.set_description("Saving scaler")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(scaler, save_path)
        pbar.update(1) # Update progress even if not saved
        
        # Step 3: Transform the features
        pbar.set_description("Transforming features")
        df_scaled[features_to_scale] = scaler.transform(df_scaled[features_to_scale])
        pbar.update(1)
    
    if save_path:
         print(f"\n\nScaler saved to: {save_path}")
            
    return df_scaled, scaler


def load_scaler(scaler_path: str) -> Any:
    """
    Loads a pre-fitted scikit-learn scaler from a file using joblib.

    Args:
        scaler_path (str): The file path to the saved scaler (.pkl or .joblib file).

    Returns:
        Any: The loaded scikit-learn scaler object.
        
    Raises:
        FileNotFoundError: If the scaler file does not exist at the given path.
    """
    print(f"Loading pre-fitted scaler from: {scaler_path}")
    try:
        scaler = joblib.load(scaler_path)
        print("✓ Scaler loaded successfully.")
        return scaler
    except FileNotFoundError:
        print(f"❌ ERROR: Scaler file not found at '{scaler_path}'.")
        raise
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred while loading the scaler: {e}")
        raise


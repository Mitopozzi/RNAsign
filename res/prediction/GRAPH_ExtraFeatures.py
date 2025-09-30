#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Create enhanced features from sequence data including rolling statistics and positional encodings.
    
    This function transforms raw sequence data into a rich feature representation by:
    1. Computing rolling window statistics for coverage and primer data
    2. Calculating gradients to capture local changes
    3. Creating relative normalized features
    4. Adding sinusoidal positional encodings at multiple frequencies
    
    Notes
    -----
    - Rolling statistics are computed with center=True for symmetric windows
    - Gradients capture local rate of change between adjacent positions
    - Relative features are normalized by max value + small epsilon for numerical stability
    - Positional encodings use varying frequencies: 1000^(i/4) for i in [0,1,2,3]
    - Missing values from rolling operations are filled with zeros
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_extra_features(df: pd.DataFrame, pos_encoding_dim: int = 4) -> tuple:
    print("The following features will be computed for each sequence:\n")
    print("  - Rolling Statistics: Mean and Standard Deviation to capture local trends and volatility.\n")
    print("  - Gradients: The difference between adjacent positions to measure local change.\n")
    print("  - Relative Normalization: Values scaled within each sequence to get local context.\n")
    print("  - Inter-Node Delta: The net change in starts/ends between adjacent nodes (net flux).\n")
    print("  - Positional Encodings: Sinusoidal signals to give the model awareness of position.\n")
    def feature_transform(group):
        """Apply feature transformations to each sequence group."""
        group = group.copy()
        
        # Rolling statistics for smoothed trends
        group['coverage_rolling_mean'] = group['coverage'].rolling(window=10, center=True, min_periods=1).mean()
        group['prime3_rolling_mean'] = group['Prime3'].rolling(window=5, center=True, min_periods=1).mean()
        group['prime5_rolling_mean'] = group['Prime5'].rolling(window=5, center=True, min_periods=1).mean()

        # NEW: Rolling standard deviation for signal volatility
        group['coverage_rolling_std'] = group['coverage'].rolling(window=10, center=True, min_periods=1).std()
        group['prime3_rolling_std'] = group['Prime3'].rolling(window=5, center=True, min_periods=1).std()
        group['prime5_rolling_std'] = group['Prime5'].rolling(window=5, center=True, min_periods=1).std()

        # Gradient features for capturing local changes
        group['coverage_gradient'] = group['coverage'].diff()
        group['Prime3_gradient'] = group['Prime3'].diff()
        group['Prime5_gradient'] = group['Prime5'].diff()
        
        # Relative normalization
        group['prime3_relative'] = group['Prime3'] / (group['Prime3'].max() + 1e-6)
        group['prime5_relative'] = group['Prime5'] / (group['Prime5'].max() + 1e-6)

        # NEW: Delta between Prime3 at node i and Prime5 at node i+1
        # This captures the transitional property between adjacent nucleotides.
        group['inter_node_primer_delta'] = group['Prime3'] - group['Prime5'].shift(-1)
        
        # Vectorized sinusoidal positional encodings
        positions = group['position'].values
        i = np.arange(pos_encoding_dim)
        angle_rates = 1 / (1000 ** (i / pos_encoding_dim))
        angle_rads = positions[:, np.newaxis] * angle_rates[np.newaxis, :]
        pe_sin = np.sin(angle_rads)
        pe_cos = np.cos(angle_rads)
        for j in range(pos_encoding_dim):
            group[f'sin_pos_{j}'] = pe_sin[:, j]
            group[f'cos_pos_{j}'] = pe_cos[:, j]
            
        # Safer imputation
        group = group.bfill().ffill()
        return group
    
    # Performance: Use .apply() with tqdm
    tqdm.pandas(desc="Creating extra features")
    grouped = df.groupby("sequence", sort=False)
    df_features = grouped.progress_apply(feature_transform, include_groups=False).reset_index(level=1, drop=True).reset_index()
    
    # Robustness: Explicitly define feature columns for the tensor
    base_features = ['coverage', 'Prime3', 'Prime5']
    generated_features = [
        'coverage_rolling_mean', 'prime3_rolling_mean', 'prime5_rolling_mean',
        'coverage_rolling_std', 'prime3_rolling_std', 'prime5_rolling_std', 
        'coverage_gradient', 'Prime3_gradient', 'Prime5_gradient',
        'prime3_relative', 'prime5_relative',
        'inter_node_primer_delta'
    ]
    pos_encoding_features = [f'{func}_pos_{j}' for j in range(pos_encoding_dim) for func in ['sin', 'cos']]
    
    final_feature_cols = base_features + generated_features + pos_encoding_features
    final_feature_cols = [col for col in final_feature_cols if col in df_features.columns]
 
    return df_features,final_feature_cols
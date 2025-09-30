#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Configures the environment for a machine learning run.

    This function sets the random seed, determines the compute device,
    creates a results directory path, changes the CWD, and calculates
    the optimal number of workers.

    Args:
        data_dir (str): The base directory for data and results.
        seed (int): The random seed for reproducibility.

    Returns:
        dict: A dictionary containing the configured parameters:
              'device', 'results_dir', 'num_workers', and 'seed'.
"""
import os
import torch
import random
import numpy as np

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_environment(seed: int = 42) -> dict:

    # 1. Set seed for reproducibility
    set_seed(seed)
    print(f"Seed set to: {seed}")

    # 2. Determine the computing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
  
    # 3. Determine the number of workers for DataLoader
    num_workers = min(max(1, os.cpu_count() * 3 // 4), 8)
    print(f"Number of workers set to: {num_workers}")
    
    return seed, device, num_workers
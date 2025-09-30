#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Creates a list of PyTorch Geometric Data objects from a processed DataFrame.

    Args:
        df_processed (pd.DataFrame): The DataFrame containing scaled features, 
                                     sequence names, and labels.
        feature_cols (list): The list of column names to use as node features.
        max_len (int, optional): The maximum sequence length to include. 
                                 Sequences longer than this will be skipped. Defaults to 150.

    Returns:
        List[Data]: A list of PyG Data objects ready for training.
"""

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from typing import List

def create_pyg_graphs_labels(
    df_processed: pd.DataFrame, 
    feature_cols: list, 
    max_len: int = 150
) -> List[Data]:
    data_list = []
    
    # Group the DataFrame by sequence name. Each group is one graph.
    grouped = df_processed.groupby("sequence")
    
    for seq_name, group in tqdm(grouped, desc="Transforming RNAs into graphs"):
        
        # Check sequence length before proceeding
        seq_len = len(group)
        if seq_len > max_len:
            continue
        
        # 1. Create the node feature tensor (x) from the specified columns
        x = torch.tensor(group[feature_cols].values, dtype=torch.float32)
        
        # 2. Get the label for the graph (y) from the 'label' column
        #    We use iloc[0] assuming the label is the same for the whole sequence.
        label = group['label'].iloc[0]
        y = torch.tensor([label], dtype=torch.long) # Standard to use 'y' for labels
        
        # 3. Create the edge index for a linear graph (connecting adjacent nodes)
        if seq_len > 1:
            indices = torch.arange(seq_len - 1, dtype=torch.long)
            src = torch.cat([indices, indices + 1])
            dst = torch.cat([indices + 1, indices])
            edge_index = torch.stack([src, dst], dim=0).contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # 4. Create the PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y, sequence_name=seq_name)
        data_list.append(data)
        
    return data_list


def create_pyg_node_labels(
    df_processed: pd.DataFrame, 
    feature_cols: list, 
    max_len: int = 150
) -> List[Data]:
    """
    Creates a list of PyG Data objects with node-specific labels.
    """
    data_list = []
    
    # Group the DataFrame by sequence name. Each group is one graph.
    grouped = df_processed.groupby("sequence")
    
    for seq_name, group in tqdm(grouped, desc="Transforming RNAs into graphs"):
        
        # Check sequence length before proceeding
        seq_len = len(group)
        if seq_len > max_len:
            continue
        
        # 1. Create the node feature tensor (x)
        x = torch.tensor(group[feature_cols].values, dtype=torch.float32)
        
        # 2. Get the node-specific labels for the graph (y)
        # Get all labels for the nodes in the current graph.
        node_labels = group['label'].values
        # Create a tensor with a label for each node. Its shape will be [num_nodes].
        y = torch.tensor(node_labels, dtype=torch.long) 
        # 3. Get the node-specific labels for the graph (y)
        node_coverage = group['coverage'].values
        # Create a tensor with the coverage for each node. Its shape will be [num_nodes].
        cov_tens = torch.tensor(node_coverage, dtype=torch.float32) 
        
        # 4. Create the edge index for a linear graph
        if seq_len > 1:
            indices = torch.arange(seq_len - 1, dtype=torch.long)
            src = torch.cat([indices, indices + 1])
            dst = torch.cat([indices + 1, indices])
            edge_index = torch.stack([src, dst], dim=0).contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # 4. Create the PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y, coverage = cov_tens, sequence_name=seq_name)
        data_list.append(data)
        
    return data_list

def create_pyg_graphs_for_prediction(df_processed: pd.DataFrame, feature_cols: list, max_len: int = 150) -> List[Data]:
    """Creates a list of PyG Data objects for prediction (no labels)."""
    data_list = []
    grouped = df_processed.groupby("sequence")
    for seq_name, group in tqdm(grouped, desc="Creating prediction graphs"):
        seq_len = len(group)
        if seq_len > max_len:
            continue
        
        x = torch.tensor(group[feature_cols].values, dtype=torch.float32)
        coverage_tensor = torch.tensor(group['coverage'].values, dtype=torch.float32)
        
        if seq_len > 1:
            indices = torch.arange(seq_len - 1, dtype=torch.long)
            src = torch.cat([indices, indices + 1])
            dst = torch.cat([indices + 1, indices])
            edge_index = torch.stack([src, dst], dim=0).contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        data = Data(x=x, edge_index=edge_index, coverage=coverage_tensor, sequence_name=seq_name)
        data_list.append(data)
    return data_list
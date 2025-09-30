#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 03:42:02 2025

@author: artho
"""
from typing import List
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.loader import DataLoader # Using DataLoader for batching is more efficient
from typing import Tuple
from typing import Any, Dict


def graph_subgraph_embeddings(
    model: torch.nn.Module,
    subgraph_list: List[Data],
    device: torch.device,
    pooling: str = 'mean'
) -> np.ndarray:
    """
    Extract sequence-level embeddings from subgraphs using a trained graph neural network.
    
    Optimized version with modern PyTorch improvements while maintaining sequential processing.
    """
    model.eval()  # Set model to evaluation mode
    subgraph_embeddings = []
    
    # Use global_mean_pool like Option One
    from torch_geometric.nn import global_mean_pool, global_max_pool
    pool_fn = global_mean_pool if pooling == 'mean' else global_max_pool
    
    # Use inference_mode and mixed precision like Option One
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=device.type=='cuda'):
        for subgraph in tqdm(subgraph_list, desc="Extracting embeddings"):
            # Move subgraph to appropriate device with non_blocking
            subgraph = subgraph.to(device, non_blocking=True)
            
            # Check if subgraph has valid structure
            if subgraph.num_nodes > 0 and subgraph.edge_index.size(1) > 0:
                # Forward pass through the model
                node_embeddings = model(subgraph.x, subgraph.edge_index)
                
                # Use PyG's optimized pooling instead of manual mean
                # Create a batch tensor of zeros (single graph)
                batch_tensor = torch.zeros(subgraph.num_nodes, dtype=torch.long, device=device)
                subgraph_emb = pool_fn(node_embeddings, batch_tensor)
                
                # Keep on GPU and collect, then transfer once at end
                subgraph_embeddings.append(subgraph_emb)
            else:
                # Handle empty subgraphs with zero embeddings
                embedding_dim = getattr(model, 'embedding_dim', node_embeddings.size(-1) if 'node_embeddings' in locals() else 64)
                subgraph_embeddings.append(torch.zeros(1, embedding_dim, device=device))
    
    # Single transfer to CPU at the end like Option One
    if subgraph_embeddings:
        return torch.cat(subgraph_embeddings, dim=0).cpu().numpy()
    else:
        return np.array([])
    
    
    
"""
    Extracts embeddings for each node from a list of subgraphs using batched processing.

    Args:
        model (torch.nn.Module): The trained graph neural network.
        subgraph_list (List[Data]): The list of PyG Data objects.
        device (torch.device): The device to run the model on.
        batch_size (int): The number of graphs to process in each batch.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - A NumPy array of all node embeddings concatenated together.
        - A NumPy array mapping each node embedding back to its original subgraph's index.
"""
    

def extract_node_embeddings(
    model: torch.nn.Module,
    subgraph_list: List[Data],
    device: torch.device,
    batch_size: int = 64  # Process in batches for efficiency
) -> Tuple[np.ndarray, np.ndarray]:

    model.eval()
    all_node_embeddings = []
    # Ensure all data objects are on the CPU before passing them to the DataLoader.
    cpu_subgraph_list = [data.to('cpu') for data in subgraph_list]
    # Use a DataLoader for efficient, parallelized batching
    loader = DataLoader(cpu_subgraph_list, batch_size=batch_size, shuffle=False)
    
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extracting node embeddings"):
            batch = batch.to(device)
            
            # Forward pass to get node embeddings for all graphs in the batch
            # Shape: [total_nodes_in_batch, embedding_dim]
            node_embs_batch = model(batch.x, batch.edge_index)
            
            # Keep on GPU for now, transfer all at once later
            all_node_embeddings.append(node_embs_batch)
            
    if not all_node_embeddings:
        return np.array([]), np.array([])

    # Concatenate all batches and move to CPU once at the end
    final_embeddings = torch.cat(all_node_embeddings, dim=0).cpu().numpy()

    # Create the mapping array from node to its original graph
    node_to_graph_map = np.concatenate([
        np.full(g.num_nodes, i) for i, g in enumerate(subgraph_list)
    ])
    
    return final_embeddings, node_to_graph_map



def extract_node_metadata(subgraph_list: List[Any]) -> Dict[str, np.ndarray]:
    """
    Extracts and flattens node-level metadata from a list of subgraph objects.
    """
    print("Extracting and concatenating node-level metadata...")
    
    # These operations are vectorized and much faster than a Python loop
    all_labels = np.concatenate([sg.y.cpu().numpy() for sg in subgraph_list])
    all_coverages = np.concatenate([sg.coverage.cpu().numpy() for sg in subgraph_list])
    all_sequences = np.concatenate([
        np.full(sg.num_nodes, sg.sequence_name) for sg in subgraph_list
    ])
    
    return {
        "labels": all_labels,
        "sequences": all_sequences,
        "coverage": all_coverages,
    }


def extract_prediction_metadata(subgraph_list: List[Any]) -> Dict[str, np.ndarray]:
    """Extracts and flattens metadata for prediction (no labels)."""
    print("Extracting and concatenating prediction metadata...")
    all_coverages = np.concatenate([sg.coverage.cpu().numpy() for sg in subgraph_list])
    all_sequences = np.concatenate([
        np.full(sg.num_nodes, sg.sequence_name) for sg in subgraph_list
    ])
    return { "sequences": all_sequences, "coverage": all_coverages }
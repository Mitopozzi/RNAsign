#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Enhanced GraphSAGE Model for RNA Sequence Analysis

This optimized model improves computational efficiency, flexibility, and stability.

Key Optimizations:
1. Vectorized/Batched Multi-Head Attention: Replaced the slow sequential loop 
   with efficient parallel processing using masking (via to_dense_batch).
2. Modularity: Introduced flexible GNNBlock and AttentionBlock for easier experimentation.
3. Modernization: Configurable normalization (e.g., GraphNorm) and activation (e.g., GELU).
4. Robustness: Automatic handling of batch vectors for normalization and attention.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import (
    SAGEConv, GATv2Conv, TransformerConv,
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm, LayerNorm, GraphNorm)
# Utility for efficient batched attention
from torch_geometric.utils import to_dense_batch
from typing import Optional, Dict, Any, Type, Tuple



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Enhanced GraphSAGE Model for RNA Sequence Analysis with Classification

This version extends the EnhancedGraphSAGE model with a classification head,
enabling it to perform direct graph-level classification tasks.
"""
# %% Helper Functions for Configuration

def get_activation(activation_name: str = 'gelu') -> nn.Module:
    """Helper to select activation function."""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
    }
    return activations.get(activation_name, nn.GELU())

def get_normalization(norm_name: str, channels: int) -> Tuple[nn.Module, bool]:
    """
    Helper to select normalization layer and determine if it requires a batch vector.
    GraphNorm and LayerNorm(mode='graph') require the batch vector.
    """
    # Note: PyG LayerNorm defaults to mode='graph'.
    norms = {
        'batch': (BatchNorm(channels), False),
        'layer': (LayerNorm(channels, mode='graph'), True), 
        'graph': (GraphNorm(channels), True),
        'none': (nn.Identity(), False),
    }
    if norm_name not in norms:
         raise ValueError(f"Unknown normalization: {norm_name}. Options: {list(norms.keys())}")
    return norms[norm_name]

def get_gnn_layer(layer_type: str = 'sage') -> Type[nn.Module]:
    """Helper to select the GNN convolution layer."""
    layers = {
        'sage': SAGEConv,
        'gatv2': GATv2Conv,
        'transformer': TransformerConv,
    }
    if layer_type not in layers:
        raise ValueError(f"Unknown GNN layer type: {layer_type}. Options: {list(layers.keys())}")
    return layers[layer_type]

# %% Modular Blocks

class GNNBlock(nn.Module):
    """
    A flexible GNN block: Conv -> Norm -> Activation -> Dropout.
    Handles residual connections and robust normalization.
    """
    def __init__(self, in_channels: int, out_channels: int, config: Dict[str, Any], use_residual: bool):
        super().__init__()
        self.layer_type = config.get('layer_type', 'sage')
        self.use_residual = use_residual
        
        GNNLayer = get_gnn_layer(self.layer_type)
        
        # Configure Convolution
        if self.layer_type == 'sage':
            self.conv = GNNLayer(in_channels, out_channels, normalize=True)
        elif self.layer_type in ['gatv2', 'transformer']:
            # Use averaging (concat=False) to maintain output dimension
            heads = config.get('gnn_heads', 4)
            self.conv = GNNLayer(in_channels, out_channels, heads=heads, dropout=config['dropout'], concat=False)
        else:
            self.conv = GNNLayer(in_channels, out_channels)

        # Configure Normalization and Activation
        self.norm, self.norm_requires_batch = get_normalization(config['normalization'], out_channels)
        self.activation = get_activation(config['activation'])
        self.dropout = nn.Dropout(config['dropout'])

        # Projection for residual connection if dimensions differ
        self.residual_proj = None
        if use_residual and in_channels != out_channels:
            # bias=False as normalization follows
            self.residual_proj = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, return_attention: bool = False) -> Any:
        x_input = x
        
        # Perform the convolution first.
        if self.layer_type == 'transformer' and return_attention:
            # If attention is requested, capture the weights.
            x, attention_weights = self.conv(x, edge_index, return_attention_weights=True)
        else:
            # Otherwise, just perform the standard convolution.
            x = self.conv(x, edge_index)
            attention_weights = None
        
        # Apply normalization robustly
        if self.norm_requires_batch:
            # Ensure batch vector is present if required (e.g., GraphNorm, LayerNorm mode='graph')
            if batch is None and not isinstance(self.norm, nn.Identity):
                 # If batch is None, assume input is a single graph (create trivial batch vector)
                 batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = self.norm(x, batch)
        else:
            # BatchNorm or Identity
            x = self.norm(x)
        
        x = self.activation(x)
        x = self.dropout(x)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj:
                x_input = self.residual_proj(x_input)
            x = x + x_input
        if attention_weights is not None:
            return x, attention_weights    
        return x

class BatchedAttentionBlock(nn.Module):
    """
    (OPTIMIZED) Efficiently applies Multi-Head Attention (MHA) to batched graphs in parallel.
    Replaces the slow sequential loop of the original implementation.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, norm_type: str, activation_name: str):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True # Essential for (B, N, D) format
        )
        # Typically LayerNorm is used in attention blocks
        self.norm, self.norm_requires_batch = get_normalization(norm_type, embed_dim)
        self.activation = get_activation(activation_name)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Note: This block assumes 'batch' is not None. The main model ensures this.
        x_input = x
        
        # 1. Convert sparse indexed representation to dense (B, N_max, D)
        x_dense, mask = to_dense_batch(x, batch)
        
        # 2. Prepare key_padding_mask for MHA
        # mask is True=Node, False=Padding. MHA expects True=Ignore (Padding).
        key_padding_mask = ~mask
        
        # 3. Apply MHA in parallel
        attn_output, _ = self.attention(
            query=x_dense, key=x_dense, value=x_dense, 
            key_padding_mask=key_padding_mask
        )
        
        # 4. Convert back to sparse indexed representation
        x_attended = attn_output[mask]
        
        # 5. Residual, Norm, and Activation (Transformer style: Add & Norm)
        x = x_attended + x_input
        
        if self.norm_requires_batch:
            x = self.norm(x, batch)
        else:
            x = self.norm(x)
            
        x = self.activation(x)
        
        return x

# %% Optimized Enhanced GraphSAGE Model

class GraphSAGE(nn.Module):
    """
    Optimized Enhanced GraphSAGE model.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 num_layers: int = 4,
                 dropout: float = 0.2,
                 use_attention: bool = True,
                 use_residual: bool = True,
                 layer_type: str = 'sage',         # 'sage', 'gatv2', 'transformer'
                 normalization: str = 'graph',     # 'batch', 'layer', 'graph', 'none' (GraphNorm is robust)
                 activation: str = 'gelu',         # 'relu', 'gelu', 'silu'
                 attention_heads: int = 4
                 ):
        super().__init__()
        
        self.config = {
            'dropout': dropout,
            'layer_type': layer_type,
            'normalization': normalization,
            'activation': activation,
            'gnn_heads': attention_heads # Used if layer_type is attention-based
        }
        self.use_attention = use_attention

        # --- GNN Layers Stack ---
        self.gnn_layers = nn.ModuleList()
        current_dim = in_channels
        for _ in range(num_layers):
            self.gnn_layers.append(
                GNNBlock(current_dim, hidden_channels, self.config, use_residual)
            )
            current_dim = hidden_channels
        
        # --- Optimized Attention Block (Optional) ---
        self.attention_block = None
        if use_attention:
            if hidden_channels % attention_heads != 0:
                 raise ValueError(f"hidden_channels ({hidden_channels}) must be divisible by attention_heads ({attention_heads})")
            
            # Use LayerNorm for the attention block normalization, as is standard
            self.attention_block = BatchedAttentionBlock(
                hidden_channels, attention_heads, dropout, 'layer', activation
            )
        
        # --- Graph-level Pooling Projection ---
        self.pool_projection = nn.Linear(hidden_channels * 3, hidden_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, return_graph_embedding: bool = False, return_attention: bool = False) -> Any:
        attention_weights = None
    
        # --- 1. GNN Stack ---
        for i, layer in enumerate(self.gnn_layers):
            # The 'layer_type' check is now inside the GNNBlock, so this loop is simpler
            if self.config.get('layer_type') == 'transformer' and i == 0 and return_attention:
                x, attention_weights = layer(x, edge_index, batch, return_attention=True)
            else:
                x = layer(x, edge_index, batch)
                
        # --- 2. Efficient Batched Attention Mechanism ---
        if self.attention_block is not None:
            if batch is None:
                current_batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            else:
                current_batch = batch
            x = self.attention_block(x, current_batch)
    
        # --- 3. Output Handling (CORRECTED LOGIC) ---
        # This new structure ensures a value is always returned.
        
        # First, determine which type of embedding to produce
        final_embedding = self._graph_pooling(x, batch) if return_graph_embedding else x
    
        # Then, decide what to return
        if return_attention:
            return final_embedding, attention_weights
        else:
            return final_embedding

    def _graph_pooling(self, x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        """Combines mean, max, and sum pooling."""
        # Ensure batch vector exists for pooling
        if batch is None:
             current_batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
             current_batch = batch

        x_mean = global_mean_pool(x, current_batch)
        x_max = global_max_pool(x, current_batch)  
        x_sum = global_add_pool(x, current_batch)
            
        x_graph = torch.cat([x_mean, x_max, x_sum], dim=1)
        x_graph = self.pool_projection(x_graph)
        return x_graph

# %% Compatibility Wrapper and Factory

def create_graphsage_model(model_type: str = "enhanced", **kwargs) -> GraphSAGE:
    """
    Factory function to create different GraphSAGE variants.
    """
    
    if model_type == "legacy_compatible":
        # Uses the compatibility wrapper
        return GraphSAGE(kwargs.get('in_channels'), kwargs.get('hidden_channels', 64))
    
    elif model_type == "enhanced":
        # Default Optimized Enhanced Configuration (SAGE + Attention Block)
        return GraphSAGE(
            in_channels=kwargs.get('in_channels'),
            hidden_channels=kwargs.get('hidden_channels', 64),
            num_layers=kwargs.get('num_layers', 4),
            use_attention=True,
            layer_type='sage',
            normalization='graph'
        )
    
    elif model_type == "transformer_style":
        # Example using TransformerConv layers
        return GraphSAGE(
            in_channels=kwargs.get('in_channels'),
            hidden_channels=kwargs.get('hidden_channels', 64),
            layer_type='transformer',
            use_attention=False # TransformerConv already includes attention
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

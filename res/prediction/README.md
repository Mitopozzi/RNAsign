# Core Prediction Modules

Here is the collection of custom scripts necessary to run RNA sign. Below is the description for the three key modules that power the pipeline: graph neural network architecture, graph construction, and embedding extraction.

## Overview

These modules form the foundation of the machine learning prediction system:

1. **SAGE_Model_Gemini.py** - GraphSAGE neural network architecture with attention mechanisms
2. **GRAPH_CreateGraphs.py** - Converts RNA sequences into graph representations
3. **GRAPH_ExtractEmbeddings.py** - Extracts features from trained models for classification

Also, before starting to use the pipeline, remember to unzip the models in /res .

```bash
unzip RF_classifier.zip
unzip TwoModels_RF_classifier.zip
```

## Module Hierarchy

```
RNA Sequence Data
    ↓
GRAPH_CreateGraphs.py → PyG Data Objects
    ↓
SAGE_Model_Gemini.py → Node Embeddings
    ↓
GRAPH_ExtractEmbeddings.py → Feature Vectors
    ↓
Classifier (XGBoost/RF/Meta-learner)
```

---

## Module 1: SAGE_Model_Gemini.py

### Purpose
Implements an optimized GraphSAGE neural network with multi-head attention for learning representations of RNA sequence structures.

### Core Model: GraphSAGE Class

**Key Features:**
- **Modular GNN layers** with support for SAGEConv, GATv2Conv, and TransformerConv
- **Multi-head attention mechanism** with efficient batched processing
- **Flexible normalization** (BatchNorm, LayerNorm, GraphNorm)
- **Residual connections** for improved gradient flow
- **Graph-level pooling** combining mean, max, and sum operations

**Configuration Parameters:**

```python
GraphSAGE(
    in_channels: int,           # Number of input features per node
    hidden_channels: int,       # Embedding dimension
    num_layers: int = 4,        # Number of GNN layers
    dropout: float = 0.2,       # Dropout rate
    use_attention: bool = True, # Enable attention block
    use_residual: bool = True,  # Enable residual connections
    layer_type: str = 'sage',   # 'sage', 'gatv2', 'transformer'
    normalization: str = 'graph', # 'batch', 'layer', 'graph', 'none'
    activation: str = 'gelu',   # can handle negative values and more stable then relu
    attention_heads: int = 4    # Number of attention heads
)
```

#### create_graphsage_model()

```python
def create_graphsage_model(model_type: str = "enhanced", **kwargs) -> GraphSAGE
```

**Available Variants:**

1. **"enhanced"** (Default)
   - SAGEConv layers + Attention block
   - GraphNorm normalization
   - Best for general RNA analysis

2. **"transformer_style"**
   - TransformerConv layers (built-in attention)
   - No separate attention block
   - Used in RNAsign production pipeline

The attention is computed but it is not used by the downstream classifiers, as no specific position or pattern is especially meaningful.
An example below of the attention on each node of the graph, colored based on the attention level.

![GNN attention example](/res/TM_Attention.png)

#### Optimization Features

1. **Xavier/Glorot Initialization** for stable training
2. **Flexible activation functions** (GELU default for smoother gradients)
3. **Robust normalization** with automatic batch handling
4. **Memory-efficient attention** using sparse-to-dense conversion
5. **Mixed precision support** via PyTorch AMP compatibility

---

## Module 2: GRAPH_CreateGraphs.py

### Purpose
Transforms processed RNA sequence data from DataFrames into PyTorch Geometric graph objects suitable for GNN training and prediction.

### Core Functions

#### 1. create_pyg_node_labels()
Creates graphs with node-level labels (for position-specific classification).

```python
def create_pyg_node_labels(
    df_processed: pd.DataFrame,
    feature_cols: list,
    max_len: int = 150
) -> List[Data]
```
**Use Case:** Training models to predict functional positions within sequences.

#### 2 create_pyg_graphs_for_prediction()
Creates graphs for inference (no labels required).

```python
def create_pyg_graphs_for_prediction(
    df_processed: pd.DataFrame,
    feature_cols: list,
    max_len: int = 150
) -> List[Data]
```
**Use Case:** Prediction pipeline on new, unlabeled data.

---

## Module 3: GRAPH_ExtractEmbeddings.py

### Purpose
Extracts embeddings and metadata from trained GNN models for downstream classification tasks.

### Core Functions

#### 1. extract_node_embeddings()
Extracts node-level embeddings efficiently using batched processing.

```python
def extract_node_embeddings(
    model: torch.nn.Module,
    subgraph_list: List[Data],
    device: torch.device,
    batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]
```

**Process:**
1. Batches graphs using `DataLoader`
2. Runs forward pass in `inference_mode` (no gradients)
3. Collects embeddings on GPU, transfers once to CPU
4. Creates mapping from nodes to their source graphs

**Returns:**
- `embeddings`: NumPy array `[total_nodes, embedding_dim]`
- `node_to_graph_map`: NumPy array `[total_nodes]` mapping nodes to graph indices

**Optimization Features:**
- Batched processing for GPU efficiency
- Single CPU transfer at end
- Automatic DataLoader parallelization
- Progress tracking with tqdm

#### 2. extract_prediction_metadata()
Extracts metadata for prediction (no labels).

```python
def extract_prediction_metadata(
    subgraph_list: List[Any]
) -> Dict[str, np.ndarray]
```

**Returns Dictionary:**
```python
{
    'sequences': np.ndarray,   # Sequence names [total_nodes]
    'coverage': np.ndarray     # Coverage values [total_nodes]
}
```
---

## Dependencies

**PyTorch Ecosystem:**
- torch >= 1.10
- torch-geometric >= 2.0
- torch-scatter, torch-sparse (PyG dependencies)

**Data Processing:**
- pandas >= 1.3
- numpy >= 1.20

**Utilities:**
- tqdm (progress bars)

---

## Citation (WIP)

If using these modules in research, cite:
- 

---

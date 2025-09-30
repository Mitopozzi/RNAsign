# RNA Signature Analysis Pipeline

A comprehensive bioinformatics pipeline for identifying functional miRNA regions from RNA sequencing coverage data using graph neural networks and machine learning.

## Overview

This pipeline consists of three integrated components:

1. **Aggregate.sh** - Combines multiple coverage tracks into a unified format
2. **Chunking.py** - Extracts stable coverage regions with flanking sequences
3. **Prediction.py** - Identifies functional miRNA regions using GNN and ensemble learning

Also, before starting to use the pipeline, remember to unzip the models in /res .

## Pipeline Workflow

```
BAM files → bedtools → Aggregate.sh → Chunking.py → Prediction.py → miRNA annotations
```

---

## Component 1: Aggregate.sh

### Purpose
Merges three separate bedtools coverage files (total, 3' prime, and 5' prime) into a single integrated coverage file for downstream analysis.

### Usage

```bash
./Aggregate.sh <total_cov.csv> <prime3_cov.csv> <prime5_cov.csv> <output_file.csv>
```

### Input Format

Three separate coverage files with format:
```
SequenceID    Position    Coverage
chr1          1           25
chr1          2           27
```

### Output Format

Combined CSV file with all coverage tracks:
```csv
RegionID,position,coverage,3prime,5prime
chr1,1,25,12,13
chr1,2,27,14,13
```

### Features

- **Intelligent merging** based on sequence ID and position
- **Missing data handling** with automatic zero-filling
- **Sorted output** by sequence and position
- **Header generation** for downstream compatibility

### Example

```bash
./Aggregate.sh total_coverage.bedcov \
               prime3_coverage.bedcov \
               prime5_coverage.bedcov \
               combined_coverage.csv
```

---

## Component 2: Chunking.py

### Purpose
Identifies and extracts genomic regions with short, high, and stable coverage patterns, providing high-quality candidate regions for new miRNA identification.

### Usage

#### Basic Command
```bash
python Chunking.py input_coverage.csv output_regions.csv
```

#### Advanced Usage
```bash
python Chunking.py combined_coverage.csv stable_regions.csv \
    --window_size 15 \
    --flank_size 30 \
    --tolerance 0.20 \
    --min_cov 10.0
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_file` | Required | Path to aggregated coverage file |
| `output_file` | Required | Path for output CSV file |
| `--window_size` | 15 | Size of the core stable region (nucleotides) |
| `--flank_size` | 30 | Size of flanking regions on each side (nucleotides) |
| `--tolerance` | 0.20 | Allowed variation percentage (0.20 = ±20%) |
| `--min_cov` | 10.0 | Minimum average coverage threshold |
| `--sep` | Auto | Field separator for input file |

### Input Format

Expects 5-column CSV from Aggregate.sh:
```csv
RegionID,position,coverage,3prime,5prime
chr1,1,25,12,13
```

### Output Format

Extracted regions with normalized positions:
```csv
RegionID,sequence,ori_position,position,coverage,3prime,5prime
chr1_region_1000_1074,chr1,1000,1,25,12,13
chr1_region_1000_1074,chr1,1001,2,26,13,13
```

### Stability Algorithm

A region is considered "stable" when:
- **Min Coverage ≥ Mean Coverage × (1 - tolerance)**
- **Max Coverage ≤ Mean Coverage × (1 + tolerance)**
- **Mean Coverage ≥ min_cov threshold**

### Features

- Rolling window analysis with configurable window size
- Overlap prevention (extracts only first stable window per block)
- Boundary validation ensuring complete region extraction
- Position normalization for cross-region comparison
- Progress tracking with tqdm

### Examples

#### High-stringency extraction
```bash
python Chunking.py input.csv output.csv --tolerance 0.10 --min_cov 20
```

#### Larger extraction windows
```bash
python Chunking.py input.csv output.csv --window_size 20 --flank_size 40
```

---

## Component 3: Prediction.py

### Purpose
Uses a pre-trained graph neural network (GNN) to generate new embeddings that will be used by pre-trained ensemble machine learning classifiers to identify functional miRNA regions from extracted stable coverage regions. The GNN consider each nucleotide in a sequence as a single node, and each region a single bi-directional graph. 

<p align="center">
  <img src="/res/Example.png" alt="Example prediction" width="600"/>
</p>


### Architecture

The pipeline uses a three-stage prediction system:

1. **Graph Neural Network (GraphSAGE)** - Generates node embeddings from coverage features
2. **Ensemble Classifiers** - XGBoost, Random Forest, or meta-learner combination
3. **Clustering & Annotation** - Annotate new miRNAs based on the results of the Ensemble Classifiers

### Usage

#### Basic Command
```bash
python Prediction.py -I stable_regions.csv -O output_dir -M XGB
```

#### With Visualization
```bash
python Prediction.py -I stable_regions.csv -O output_dir -M TM -P
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-I, --input` | Input coverage file from Chunking.py (CSV format) |
| `-O, --output_dir` | Output directory for predictions and plots |
| `-M, --model_type` | Classifier model: `XGB`, `RF`, or `TM` (Two Models ensemble) |

### Optional Arguments

| Argument | Description |
|----------|-------------|
| `-P, --plot` | Generate visualization plots of prediction probabilities |

### Model Types

#### XGB (XGBoost)
- Fast inference
- Single model predictions
- Good for large datasets

#### RF (Random Forest)
- Robust predictions
- Single model
- Handles noisy data well

#### TM (Two Models)
- **Recommended for best accuracy**
- Combines RF and XGB predictions via meta-learner
- Produces ensemble predictions with higher accuracy
- Outputs both individual and meta-model predictions

### Input Format

Expects output from Chunking.py with columns:
```csv
sequence,position,coverage,Prime3,Prime5
chr1_region_1000_1074,1,25,12,13
```

### Output Files

#### 1. Prediction Results
- **Single Models (XGB/RF)**: `PRED_{filename}_Predictions_{MODEL}_Summary.csv`
- **Two Models (TM)**: 
  - `PRED_{filename}_Predictions_TM.csv` (detailed with individual model predictions)
  - `PRED_{filename}_Predictions_TM_Summary.csv` (meta-model predictions only)

Format:
```csv
sequences,PositionInGraph,PredictedLabel,PredictionProbability
chr1_region_1000_1074,1,0,0.23
chr1_region_1000_1074,2,1,0.87
```

#### 2. Cluster Annotations
`PRED_{filename}_ClustAnnotation.saf`

SAF format for functional miRNA regions:
```
GeneID  Chr     Start   End     Strand
cluster_1_chr1  chr1    1000    1029    +
```

#### 3. Visualization (Optional, with -P flag)
`PRED_{filename}_Probability_Plots.pdf`

Generate a few example candidates showing prediction probabilities across sequences.

### Features

**Trained Features**
- 23 input features per node (3 base + 12 statistical + 8 positional encoding)
- Rolling window statistics
- Positional encoding with sine/cosine functions
- Standardized scaling using pre-trained scaler

**Graph Construction**
- Nodes represent nucleotide positions
- Edges connect neighboring positions
- Maximum sequence length: 150 nucleotides
- Batch processing for memory efficiency

**Ensemble Prediction (TM Mode)**
- Combines Random Forest and XGBoost predictions
- Meta-learner (Logistic Regression) for final predictions
- Provides confidence scores from multiple models

**Clustering Algorithm**
- Minimum cluster size: 15 nucleotides (default)
- Tolerance: 2 non-functional positions allowed within cluster
- Outputs in SAF format for downstream analysis

### Model Requirements

Models must be located in `res/` directory with the following structure:

```
res/
├── best_model_XGB.pth              # GNN weights for XGB
├── XGB_classifier.joblib           # XGBoost classifier
├── scaler_XGB.pkl                  # Feature scaler for XGB
├── best_model_RF.pth               # GNN weights for RF
├── RF_classifier.joblib            # Random Forest classifier
├── scaler_RF.pkl                   # Feature scaler for RF
├── best_model_TwoModels.pth        # GNN weights for TM
├── TwoModels_RF_classifier.joblib  # RF classifier for TM
├── TwoModels_XGB_classifier.joblib # XGB classifier for TM
├── MetaLearner_LR_TwoModels.joblib # Meta-learner for TM
└── scaler_TwoModels.pkl            # Feature scaler for TM
```

### Examples

#### Quick prediction with XGBoost
```bash
python Prediction.py -I regions.csv -O results/ -M XGB
```

#### Best accuracy with ensemble and plots
```bash
python Prediction.py -I regions.csv -O results/ -M TM -P
```

#### Random Forest with visualization
```bash
python Prediction.py -I regions.csv -O results/ -M RF -P
```

---

## Complete Pipeline Example

### Step 1: Generate Coverage Files
```bash
# Using bedtools to generate coverage from BAM files
bedtools genomecov -d -ibam sample.bam > total_coverage.bedcov
bedtools genomecov -d -ibam sample.bam -5 > prime5_coverage.bedcov
bedtools genomecov -d -ibam sample.bam -3 > prime3_coverage.bedcov
```

### Step 2: Aggregate Coverage Tracks
```bash
./Aggregate.sh total_coverage.bedcov \
               prime3_coverage.bedcov \
               prime5_coverage.bedcov \
               combined_coverage.csv
```

### Step 3: Extract Stable Regions
```bash
python Chunking.py combined_coverage.csv \
                   stable_regions.csv \
                   --window_size 15 \
                   --flank_size 30 \
                   --tolerance 0.20 \
                   --min_cov 10.0
```

### Step 4: Predict Functional Regions
```bash
python Prediction.py -I stable_regions.csv \
                     -O predictions/ \
                     -M TM \
                     -P
```

### Expected Output Structure
```
predictions/
├── PRED_stable_regions_Predictions_TM.csv
├── PRED_stable_regions_Predictions_TM_Summary.csv
├── PRED_stable_regions_ClustAnnotation.saf
└── PRED_stable_regions_Probability_Plots.pdf
```

---

## Citation (WIP)

If you use this pipeline in your research, please cite appropriately based on your institution's guidelines.

---


#!/usr/bin/env python3
"""
Script to identify clusters of positions with label 1 and output SAF format for DESeq2.

The script:
1. Identifies strings with at least 10 positions labeled as 1
2. Clusters consecutive label 1 positions
3. Extends clusters until the first label 0 (with configurable tolerance for 0s)
4. Outputs results in SAF format compatible with DESeq2
"""

import pandas as pd
import argparse
import sys
from typing import List, Tuple

def find_label1_clusters(df: pd.DataFrame, min_cluster_size: int = 10, 
                        tolerance: int = 0) -> List[Tuple[str, int, int, int]]:
    """
    Find clusters of label 1 positions and extend them until first label 0.
    
    Args:
        df: DataFrame with columns ['SequenceName', 'ori_pos', 'PredictedLabel']
        min_cluster_size: Minimum number of label 1 positions to form a cluster
        tolerance: Number of consecutive 0s to ignore when extending clusters
    
    Returns:
        List of tuples (sequence_name, start_pos, end_pos, num_ones)
    """
    clusters = []

    for seq_name, group in df.groupby('SequenceName'):
        group = group.sort_values('ori_pos').reset_index(drop=True)
        positions = group['ori_pos'].to_numpy()
        labels = group['PredictedLabel'].to_numpy()

        i = 0
        while i < len(positions):
            if labels[i] == 1:
                # Start of a potential cluster
                cluster_start = positions[i]
                ones_count = 1
                j = i + 1
                
                # Collect consecutive 1s and 0s within tolerance
                consecutive_zeros = 0
                last_included = i
                
                while j < len(positions):
                    if labels[j] == 1:
                        ones_count += 1
                        consecutive_zeros = 0
                        last_included = j
                    else:  # label == 0
                        consecutive_zeros += 1
                        if consecutive_zeros > tolerance:
                            break
                        last_included = j
                    j += 1
                
                # Check if we have enough 1s to form a cluster
                if ones_count >= min_cluster_size:
                    cluster_end = positions[last_included]
                    clusters.append((seq_name, cluster_start, cluster_end, ones_count))
                
                # Move past this cluster
                i = last_included + 1
            else:
                i += 1

    return clusters

def create_saf_output(clusters: List[Tuple[str, int, int, int]], 
                     output_file: str) -> None:
    """
    Create SAF format output file for DESeq2.
    
    Args:
        clusters: List of (sequence_name, start_pos, end_pos, num_ones) tuples
        output_file: Output filename
    """
    with open(output_file, 'w') as f:
        # Write SAF header
        f.write("GeneID\tChr\tStart\tEnd\tStrand\n")
        
        # Write each cluster as a named region
        last_seq = None
        cluster_idx = 0
        for seq_name, start_pos, end_pos, num_ones in clusters:
            if seq_name != last_seq:
                cluster_idx = 1
                last_seq = seq_name
            else:
                cluster_idx += 1
            gene_id = f"{seq_name}_cluster_{cluster_idx}"
            chr_name = seq_name.split("_region")[0]
            # SAF format: GeneID, Chr, Start, End, Strand - STRAND is hardcorded as +
            f.write(f"{gene_id}\t{chr_name}\t{start_pos}\t{end_pos}\t+\n")

def validate_input_file(filename: str) -> bool:
    """
    Validate that input file has required columns and valid values.
    
    Args:
        filename: Input file path
    
    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_csv(filename, sep=',', nrows=10)
        required_cols = ['SequenceName', 'ori_pos', 'PredictedLabel']
        if not all(col in df.columns for col in required_cols):
            print("Error: Input file missing required columns:")
            print(f"Required: {required_cols}")
            print(f"Found: {list(df.columns)}")
            return False
        if not set(df['PredictedLabel'].unique()).issubset({0, 1}):
            print("Error: PredictedLabel column must contain only 0 or 1")
            return False
        return True
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Identify label 1 clusters and create SAF format output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i input.tsv -o output.saf
  %(prog)s -i input.tsv -o output.saf --min-cluster-size 15 --tolerance 2
  
SAF Format Output:
The output file will contain columns: GeneID, Chr, Start, End, Strand
Compatible with DESeq2 featureCounts analysis.
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input comma-separated CSV file with predicted labels')
    parser.add_argument('-o', '--output', required=True,
                       help='Output SAF file')
    parser.add_argument('--min-cluster-size', type=int, default=10,
                       help='Minimum number of label 1 positions for a cluster (default: 10)')
    parser.add_argument('--tolerance', type=int, default=0,
                       help='Number of consecutive 0s to ignore when extending clusters (default: 0)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information about clusters found')
    
    args = parser.parse_args()
    
    # Validate input file
    if not validate_input_file(args.input):
        print("Error: Input file validation failed.")
        sys.exit(1)
    
    try:
        # Read input file
        print(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, sep=',')
        
        print(f"Loaded {len(df)} rows from {df['SequenceName'].nunique()} sequences")
        
        # Find clusters
        print(f"Finding clusters with min size {args.min_cluster_size} and tolerance {args.tolerance}")
        clusters = find_label1_clusters(df, args.min_cluster_size, args.tolerance)
        
        if not clusters:
            print("No clusters found matching the criteria.")
            sys.exit(0)
        
        print(f"Found {len(clusters)} clusters")
        
        if args.verbose:
            print(f"\nFound {len(clusters)} clusters:")
            for i, (seq_name, start_pos, end_pos, num_ones) in enumerate(clusters, 1):
                cluster_size = end_pos - start_pos + 1
                print(f"  Cluster {i}: {seq_name} positions {start_pos}-{end_pos} (span: {cluster_size}, ones: {num_ones})")
        
        # Create SAF output
        print(f"Writing SAF output to: {args.output}")
        create_saf_output(clusters, args.output)
        
        print("Done!")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
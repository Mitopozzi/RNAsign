#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys
from typing import List

def process_sequence_group(group: pd.DataFrame, stable_size: int, flank_size: int, stability_pct: float, min_coverage_threshold: float) -> List[pd.DataFrame]:
    """
    Processes a single sequence (e.g., chromosome) to find and extract stable regions.
    
    This function relies on positional indexing (iloc) for robustness. It assumes the 
    input 'group' is sorted by position.
    """
    target_len = stable_size + 2 * flank_size
    seq_len = len(group)
    
    if seq_len < target_len:
        return []

    # 1. Calculate rolling statistics (window=stable_size)
    # center=False (default) ensures the result is assigned to the right edge of the window.
    rolling_mean = group['coverage'].rolling(window=stable_size).mean()
    rolling_min = group['coverage'].rolling(window=stable_size).min()
    rolling_max = group['coverage'].rolling(window=stable_size).max()

    # 2. Define stability criteria based on the percentage threshold
    # Stability means Min >= Mean*(1-pct) AND Max <= Mean*(1+pct)
    lower_bound = rolling_mean * (1 - stability_pct)
    upper_bound = rolling_mean * (1 + stability_pct)

    # 3. Identify stable indices (the right edge of the stable window)
    is_stable = (
        (rolling_min >= lower_bound) &
        (rolling_max <= upper_bound) &
        (rolling_mean >= min_coverage_threshold)
    )
    
    # 4. Handle the specific case where Mean=0. 
    # If Mean=0, Min and Max must be 0. The bounds calculation might result in NaNs if Mean is 0.
    # We use .mask() to explicitly define stability when Mean is 0 (Max must also be 0).
    # np.isclose is used for robust comparison with zero in case of float coverage.
    is_stable = is_stable.mask(np.isclose(rolling_mean, 0), np.isclose(rolling_max, 0))

    # 5. Handle Overlaps: Identify the START of contiguous stable blocks
    # We only want to extract one 100nt region per stable block.
    # A block starts if the current window is stable AND the previous window was NOT stable.
    is_stable_series = is_stable.astype(bool)
    block_start = is_stable_series & (~is_stable_series.shift(1, fill_value=False))

    # Get the integer indices (iloc positions) where a block starts.
    # This is the END index of the first stable 20nt window in the block.
    # We use np.where() to get the iloc positions, which is robust regardless of the DataFrame index.
    stable_end_indices = np.where(block_start)[0]

    extracted_regions = []

    # 6. Process and Extract stable regions
    for end_idx in stable_end_indices:
        # Calculate the start iloc of the stable region
        start_idx = end_idx - stable_size + 1

        # Calculate the extraction boundaries (including flanks)
        extract_start_iloc = start_idx - flank_size
        # extract_end_iloc is the inclusive index for the end of the extraction zone
        extract_end_iloc = end_idx + flank_size 

        # 7. Boundary Checks
        # Ensure we can extract the full 100nt region
        if extract_start_iloc < 0 or extract_end_iloc >= seq_len:
            continue

        # 8. Extraction using .iloc for positional slicing
        # We slice from extract_start up to extract_end + 1 (exclusive upper bound for iloc)
        region_data = group.iloc[extract_start_iloc : extract_end_iloc + 1].copy()
        
        # Sanity check for continuity (if input data had gaps, length might be wrong)
        if len(region_data) != target_len:
            continue

        # 9. Metadata and Renaming
        seq_name = group['sequence'].iloc[0]
        # Create a unique name based on the coordinates of the extracted region
        original_start_pos = region_data['position'].iloc[0]
        original_end_pos = region_data['position'].iloc[-1]
        
        # Ensure position columns are integers for the ID if they were loaded as floats
        try:
            start_pos_int = int(original_start_pos)
            end_pos_int = int(original_end_pos)
            new_seq_name = f"{seq_name}_region_{start_pos_int}_{end_pos_int}"
        except ValueError:
            # Fallback if positions are not standard integers
            new_seq_name = f"{seq_name}_region_start{original_start_pos}"

        region_data['RegionID'] = new_seq_name
        
        # Rename original position to 'ori_position'
        region_data.rename(columns={'position': 'ori_position'}, inplace=True)
        
        # Add normalized position (1 to 100)
        region_data['position'] = range(1, target_len + 1)

        extracted_regions.append(region_data)
            
    return extracted_regions

def identify_and_extract_stable_regions(
    df: pd.DataFrame,
    stable_size: int = 15,
    flank_size: int = 30,
    stability_pct: float = 0.20,
    min_coverage_threshold: float = 10.0
) -> pd.DataFrame:
    """
    Main function to identify stable coverage regions and extract the surrounding context.
    """
    
    print("\nStarting extraction of stable regions...")
    print(f"Criteria: Stable Size={stable_size}nt, Flank Size={flank_size}nt, Variance=+/-{stability_pct*100:.0f}%, Min Avg Cov>={min_coverage_threshold}")

    # Ensure required columns exist
    if not all(col in df.columns for col in ['sequence', 'position', 'coverage']):
        print("Error: DataFrame must contain 'sequence', 'position', and 'coverage' columns.")
        return pd.DataFrame()
        
    # Ensure data is sorted correctly before processing (Crucial for rolling operations)
    print("Sorting input data...")
    df_sorted = df.sort_values(by=['sequence', 'position'])

    # Group by sequence and process independently. sort=False because we already sorted.
    grouped = df_sorted.groupby('sequence', sort=False)
    
    all_extracted_data = []
    
    for seq_name, group in tqdm(grouped, desc="Analyzing sequences"):
        # We pass the group (which has group.name set by pandas groupby)
        extracted = process_sequence_group(group, stable_size, flank_size, stability_pct, min_coverage_threshold)
        if extracted:
            all_extracted_data.extend(extracted)

    # Combine results
    if not all_extracted_data:
        print("\nNo stable regions meeting the criteria were found.")
        return pd.DataFrame()

    print("\nConcatenating results...")
    results_df = pd.concat(all_extracted_data, ignore_index=True)
    
    # Reorder final columns for clarity
    # Identify original columns other than the main three
    extra_cols = [col for col in df.columns if col not in ['sequence', 'position', 'coverage']]
    # Final desired order
    final_columns = ['RegionID', 'sequence', 'ori_position', 'position', 'coverage'] + extra_cols
    
    # Ensure all expected columns exist before reordering (safety check)
    final_columns = [col for col in final_columns if col in results_df.columns]
    results_df = results_df[final_columns]
    
    print(f"\nSuccessfully extracted {results_df['RegionID'].nunique()} stable regions.")
    return results_df

def load_and_prepare_data(input_file, sep):
    """
    Loads coverage data, attempting to identify common formats (e.g., bedtools output).
    """
    print(f"Loading data from {input_file}...")
    try:
        # Read the file assuming no header.
        df = pd.read_csv(input_file, sep=sep, header=None, comment='#', engine='python')

        if df.shape[1] < 3:
            raise ValueError("Unsupported file format. Expected at least 3 columns (SeqID, Position, Coverage).")

        # Standardize column names based on common formats
        if df.shape[1] == 3:
            # Assuming Format: SeqID, Position, Coverage.
            print("Detected 3 columns. Assuming format: SeqID, Position, Coverage.")
            df.columns = ['sequence', 'position', 'coverage']
        elif df.shape[1] >= 4:
            # Assuming BED/bedcov format: SeqID, Start, End, Coverage, [Extra...]
            print("Detected more columns. Assuming format: SeqID, Position, Coverage, 3prime, 5prime")
            df.columns = ['sequence', 'position', 'coverage','3prime','5prime']
        # Ensure data types are appropriate
        # Allow float for position and coverage for flexibility, but handle NaNs
        df['position'] = pd.to_numeric(df['position'], errors='coerce')
        df['coverage'] = pd.to_numeric(df['coverage'], errors='coerce')
        df.dropna(subset=['position', 'coverage'], inplace=True)
        
        return df

    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return None

# Command Line Interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract stable coverage regions (e.g., 20nt, +/- 20%% variation) and flanking context (e.g., 40nt) from coverage data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("input_file", help="Path to the input coverage file (e.g., bedtools output). Auto-detects 3 or 4 column formats.")
    parser.add_argument("output_file", help="Path for the output CSV file.")
    parser.add_argument("--sep", default=None, help="Separator for the input file.")
    parser.add_argument("--min_cov", type=float, default=10.0, help="Minimum average coverage threshold. Set to 0 to include stable zeros.")
    parser.add_argument("--tolerance", type=float, default=0.20, help="Allowed variation percentage relative to the mean (0.20 means +/- 20%%).")
    parser.add_argument("--window_size", type=int, default=15, help="Size of the core stable region.")
    parser.add_argument("--flank_size", type=int, default=30, help="Size of the flanking regions (upstream and downstream).")

    if len(sys.argv) < 3:
        parser.print_help()
        print("\nError: Please provide input and output file paths.")
        print("Example: python extract_stable_regions.py input.bedcov output.csv\n")
        sys.exit(1)

    args = parser.parse_args()

    # 1. Load Data
    df_input = load_and_prepare_data(args.input_file, args.sep)
    if df_input is None:
        sys.exit(1)

    # 2. Process Data
    df_results = identify_and_extract_stable_regions(
        df_input,
        stable_size=args.window_size,
        flank_size=args.flank_size,
        stability_pct=args.tolerance,
        min_coverage_threshold=args.min_cov
    )

    # 3. Save Results
    if not df_results.empty:
        df_results.to_csv(args.output_file, index=False)
        print(f"\nResults saved to {args.output_file}")
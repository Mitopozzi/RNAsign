#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Loads data and checks for a common separator issue.

    Args:
        file_path (str): The path to the data file.
        **kwargs: Keyword arguments passed to pd.read_csv.

    Returns:
        tuple[pd.DataFrame | None, bool]: A tuple containing:
            - The loaded DataFrame, or None if an error occurred.
            - A boolean flag which is True if a single-column warning was issued.
"""
import pandas as pd
from pandas.errors import ParserError
from typing import Tuple, Optional
import sys

def load_data(file_path: str, **kwargs) -> tuple[pd.DataFrame | None, bool]:

    try:
        df = pd.read_csv(file_path, **kwargs)
        
        # Check if the DataFrame has 1 or fewer columns.
        if df.shape[1] <= 1:
            print("⚠️ Warning: The loaded DataFrame has only one column.")
            print(f"   This may indicate an incorrect separator for the file '{file_path}'.")
            # Return the DataFrame but signal that a warning was issued.
            return df, True

        # If the check passes, return the DataFrame and False for the warning flag.
        return df, False

    except FileNotFoundError:
        print(f"❌ Error: File not found at '{file_path}'")
        sys.exit(1)
    except ParserError:
        print(f"❌ Error: Could not parse the file at '{file_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    


def load_and_prepare_data(
    file_path: str, **kwargs
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], bool]:
    """
    Loads data and returns two versions: a full DataFrame with headers and a
    core DataFrame with specific columns.

    Args:
        file_path (str): The path to the data file.
        **kwargs: Keyword arguments passed to pd.read_csv (e.g., sep='\t').

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None, bool]: A tuple containing:
            - The full DataFrame with all columns and headers.
            - The core DataFrame with selected columns.
            - A boolean flag which is True if a single-column warning was issued.
    """
    try:
        # 1. Load the full DataFrame, assuming it has a header
        df_full = pd.read_csv(file_path, **kwargs)
        
        # Check for a potential separator issue
        warning_issued = False
        if df_full.shape[1] <= 1:
            print("⚠️ Warning: The loaded DataFrame has only one column.")
            print(f"   This may indicate an incorrect separator for '{file_path}'.")
            warning_issued = True
            sys.exit(1)

        # 2. Define and create the core DataFrame with specific columns
        core_columns = ['RegionID', 'position', 'coverage', '3prime', '5prime']
        
        # Check if the required columns exist in the loaded data
        if not all(col in df_full.columns for col in core_columns):
            missing = set(core_columns) - set(df_full.columns)
            print(f"❌ Error: The file is missing required columns: {list(missing)}")
            sys.exit(1)
            return None, None, False
            
        df = df_full[core_columns]

        return df_full, df, warning_issued

    except FileNotFoundError:
        print(f"❌ Error: File not found at '{file_path}'")
        sys.exit(1)
        return None, None, False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
        return None, None, False
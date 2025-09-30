#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Calculates and plots a normalized confusion matrix and ROC curve for a given
    set of predictions.

    Args:
        y_true (np.ndarray): The array of true labels.
        y_pred (np.ndarray): The array of predicted labels.
        y_proba (np.ndarray): The array of predicted probabilities for the positive class.
        save_path (str, optional): The file path to save the plot. Defaults to None.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import numpy as np

def plot_evaluation_metrics(y_true, y_pred, y_proba, save_path: str = None):
    print("\n--- GENERATING ROC AND CONFUSION MATRIX ---")
    
    # --- 1. Calculate Metrics Internally ---
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    # --- 2. Create Plots ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_style("whitegrid")

    # Normalized Confusion Matrix
    cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Non-Functional (0)', 'Functional (1)'],
                yticklabels=['Non-Functional (0)', 'Functional (1)'], ax=ax[0])
    ax[0].set_xlabel('Predicted Label')
    ax[0].set_ylabel('True Label')
    ax[0].set_title(f"Normalized Confusion Matrix\nAccuracy: {accuracy:.4f}")

    # ROC Curve
    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate (Recall)')
    ax[1].set_title('Receiver Operating Characteristic (ROC)')
    ax[1].legend(loc="lower right")

    plt.tight_layout()

    # --- 3. Save the Plot ---
    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✓ Evaluation plots saved to {save_path}")
        except Exception as e:
            print(f"❌ Error saving evaluation plots: {e}")
    
    plt.show()
    
"""
    Generates predictions for a full dataset, creates a detailed report,
    prints a summary, and saves the report to a CSV file.

    Args:
        classifier (Any): A trained scikit-learn classifier with .predict() and .predict_proba() methods.
        X (np.ndarray): The full feature matrix for all nodes.
        y_true (np.ndarray): The full array of true labels for all nodes.
        metadata (Dict[str, np.ndarray]): A dictionary containing node-level metadata.
                                          Expected keys: 'sequences', 'coverage'.
        save_path (str, optional): The file path to save the CSV report. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the detailed prediction results.
"""

import pandas as pd
from typing import Dict, Any

def generate_prediction_report(
    classifier: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    metadata: Dict[str, np.ndarray],
    save_path: str = None
) -> pd.DataFrame:
 
    print("\n--- ANNOTATING NODES ---")

    # 1. Generate predictions for the ENTIRE dataset
    print(f"Generating predictions for all {len(X)} nodes...")
    y_pred = classifier.predict(X)
    y_proba = classifier.predict_proba(X)[:, 1]

    # 2. Create a comprehensive DataFrame with the results
    results_df = pd.DataFrame({
        'SequenceName': metadata['sequences'],
        'Coverage': metadata['coverage'],
        'TrueLabel': y_true,
        'PredictedLabel': y_pred,
        'PredictionProbability': y_proba
    })

    # 3. Add a column to check correctness and calculate summary
    results_df['IsCorrect'] = (results_df['TrueLabel'] == results_df['PredictedLabel'])
    overall_accuracy = results_df['IsCorrect'].mean()
    
    print("--- Overall Performance Summary (Full Dataset) ---")
    print(f"Overall Accuracy: {overall_accuracy*100:.4f}%")

    # 4. Save the DataFrame if a path is provided
    if save_path:
        try:
            results_df.to_csv(save_path, index=False)
            print(f"✓ Full prediction report saved to {save_path}")
        except Exception as e:
            print(f"❌ Error saving prediction report: {e}")
            
    return results_df


def save_predictions_to_csv(
    metadata: Dict[str, np.ndarray],
    save_path: str
):
    """
    Creates a DataFrame from predictions, adds an 'ori_pos' column by
    parsing the 'SequenceName', and saves the result to a CSV file.
    """
    print(f"Preparing to save {len(metadata['sequences'])} predictions...")
    
    # 1. Create the initial DataFrame from the metadata
    results_df = pd.DataFrame({
        'SequenceName': metadata['sequences'],
        'Scaled Coverage': metadata['coverage'], # 
        'PredictedLabel': metadata['PredictedLabel'],
        'PredictionProbability': metadata['PredictionProbability']
    })

    # 2. Add the 'ori_pos' column
    print("Calculating original positions from sequence names...")
    try:
        # Vectorized step 1: Extract the start position for each row.
        # It splits '..._region_35_109' by '_' and takes the second-to-last part ('35').
        start_positions = results_df['SequenceName'].str.split('_').str[-2].astype(int)

        # Vectorized step 2: Calculate the relative position (0, 1, 2, ...) within each group.
        within_group_positions = results_df.groupby('SequenceName').cumcount()

        # Vectorized step 3: Add them to get the absolute original position.
        results_df['ori_pos'] = start_positions + within_group_positions
        
        # Reorder columns to make the output clearer
        final_cols = ['SequenceName', 'ori_pos', 'Scaled Coverage', 'PredictedLabel', 'PredictionProbability']
        results_df = results_df[final_cols]

    except (ValueError, IndexError) as e:
        print(f"⚠️ Warning: Could not calculate 'ori_pos'. The 'SequenceName' format may be unexpected. Error: {e}")

    # 3. Save the final DataFrame
    try:
        results_df.to_csv(save_path, index=False)
        print(f"✓ Predictions with original positions saved to {save_path}")
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")
        
def save_Triple_predictions_to_csv(
    metadata: Dict[str, np.ndarray],
    save_path: str
):
    """
    Creates a DataFrame from predictions, adds an 'ori_pos' column by
    parsing the 'sequences', and saves the result to a CSV file.
    """
    print(f"Preparing to save {len(metadata['sequences'])} predictions...")
    
    # 1. Create the initial DataFrame from the metadata
    results_df = pd.DataFrame({
        'SequenceName': metadata['sequences'],
        'Coverage': metadata['coverage'],  # Position within window
        'TrueLabel': metadata['labels'],
        'PredictedLabel': metadata['PredictedLabel'],
        'RF_Probability': metadata['RF_Probability'],
        'XGB_Probability': metadata['XGB_Probability'],
        'Meta_Probability': metadata['Meta_Probability'],
        'PredictionProbability': metadata['PredictionProbability']
    })
    
    # 2. Add the 'ori_pos' column
    print("Calculating original positions from sequence names...")
    try:
        # Vectorized step 1: Extract the start position for each row.
        # It splits '..._region_35_109' by '_' and takes the second-to-last part ('35').
        start_positions = results_df['SequenceName'].str.split('_').str[-2].astype(int)
        
        # Vectorized step 2: Since you already have positions within windows, 
        # we can calculate original position as: start_position + position - 1
        # (subtract 1 because your positions are 1-based but we want 0-based offset)
        results_df['ori_pos'] = start_positions + results_df['Position'] - 1
        
        # Reorder columns to make the output clearer
        final_cols = [
            'SequenceName', 
            'ori_pos', 
            'Position',  # Keep the window position for reference
            'TrueLabel',
            'PredictedLabel', 
            'RF_Probability',
            'XGB_Probability', 
            'Meta_Probability',
            'PredictionProbability'
        ]
        results_df = results_df[final_cols]
        
    except (ValueError, IndexError) as e:
        print(f"⚠️ Warning: Could not calculate 'ori_pos'. The 'SequenceName' format may be unexpected. Error: {e}")
        # If ori_pos calculation fails, still save what we have
        
    # 3. Save the final DataFrame
    try:
        results_df.to_csv(save_path, index=False)
        print(f"✓ Predictions with original positions saved to {save_path}")
        print(f"CSV contains {len(results_df)} rows with columns: {list(results_df.columns)}")
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")


def save_multipredictions_to_csv(
    metadata: Dict[str, np.ndarray],
    save_path: str
):
    """
    Creates a DataFrame from predictions, adds an 'ori_pos' column by
    parsing the 'SequenceName', and saves the result to a CSV file.
    
    Handles both single model predictions and Two Models predictions with individual model results.
    """
    print(f"Preparing to save {len(metadata['sequences'])} predictions...")
    
    # 1. Create the initial DataFrame - start with common columns
    results_df = pd.DataFrame({
        'SequenceName': metadata['sequences'],
        'Scaled Coverage': metadata['coverage'],
        'PredictedLabel': metadata['PredictedLabel'],
        'PredictionProbability': metadata['PredictionProbability']
    })
    
    # 2. Add individual model predictions if they exist (Two Models case)
    if 'RF_PredictedLabel' in metadata:
        results_df['RF_PredictedLabel'] = metadata['RF_PredictedLabel']
        results_df['RF_PredictionProbability'] = metadata['RF_PredictionProbability']
        print("Added RF individual model predictions to output...")
        
    if 'XGB_PredictedLabel' in metadata:
        results_df['XGB_PredictedLabel'] = metadata['XGB_PredictedLabel']
        results_df['XGB_PredictionProbability'] = metadata['XGB_PredictionProbability']
        print("Added XGB individual model predictions to output...")
    
    # 3. Add the 'ori_pos' column
    print("Calculating original positions from sequence names...")
    try:
        # Vectorized step 1: Extract the start position for each row.
        # It splits '..._region_35_109' by '_' and takes the second-to-last part ('35').
        start_positions = results_df['SequenceName'].str.split('_').str[-2].astype(int)
        # Vectorized step 2: Calculate the relative position (0, 1, 2, ...) within each group.
        within_group_positions = results_df.groupby('SequenceName').cumcount()
        # Vectorized step 3: Add them to get the absolute original position.
        results_df['ori_pos'] = start_positions + within_group_positions
        
        # 4. Reorder columns to make the output clearer
        # Start with basic columns, then add individual model columns if they exist
        basic_cols = ['SequenceName', 'ori_pos', 'Scaled Coverage', 'PredictedLabel', 'PredictionProbability']
        
        # Add individual model columns if they exist
        individual_cols = []
        if 'RF_PredictedLabel' in results_df.columns:
            individual_cols.extend(['RF_PredictedLabel', 'RF_PredictionProbability'])
        if 'XGB_PredictedLabel' in results_df.columns:
            individual_cols.extend(['XGB_PredictedLabel', 'XGB_PredictionProbability'])
            
        final_cols = basic_cols + individual_cols
        results_df = results_df[final_cols]
        
    except (ValueError, IndexError) as e:
        print(f"⚠️ Warning: Could not calculate 'ori_pos'. The 'SequenceName' format may be unexpected. Error: {e}")
    
    # 5. Save the final DataFrame
    try:
        results_df.to_csv(save_path, index=False)
        num_cols = len(results_df.columns)
        if num_cols > 5:
            print(f"✓ Two Models predictions ({num_cols} columns) with original positions saved to {save_path}")
        else:
            print(f"✓ Single model predictions with original positions saved to {save_path}")
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction pipeline for the identification of miRNAs from transcriptional signature

Main pipeline of RNA sign
"""
# %% PATH DETERMINATION 
# Get the absolute path of the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the project's root directory by going one level up from the /bin directory
project_root = os.path.dirname(script_dir)
# Define the base path for scripts and the assumed base path for models
SCRIPTS_PATH = os.path.join(project_root, "res", "prediction")
# Define the directory where the models (GNN weights, Scaler, Classifier) are stored
MODEL_BASE_DIR = os.path.join(project_root, "res")


# %% IMPORTS - Core Libraries
import argparse
import os
import sys
import time
import gc
import numpy as np
from functools import partial

try:
    import torch
    from tqdm import tqdm
    from torch_geometric.loader import DataLoader
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Please ensure torch, torch_geometric, pandas, tqdm, matplotlib, and seaborn are installed.")
    sys.exit(1)

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# %% IMPORTS - Internal Scripts

if SCRIPTS_PATH not in sys.path:
    sys.path.append(SCRIPTS_PATH)

try:
    from MISC_SeedEnvConfig import setup_environment
    from MISC_LoadData import load_and_prepare_data
    from GRAPH_ExtraFeatures import create_extra_features
    from GRAPH_Scaler import load_scaler
    from GRAPH_CreateGraphs import create_pyg_graphs_for_prediction
    from SAGE_Model_Gemini import create_graphsage_model
    from GRAPH_ExtractEmbeddings import extract_prediction_metadata
    from GRAPH_ClassifierPlot import save_multipredictions_to_csv
    from GRAPH_LoadModels import load_prediction_model, load_classifier
    from MISC_ClusterLabels import find_label1_clusters, create_saf_output

except ImportError as e:
    print(f"Error importing internal modules from {SCRIPTS_PATH}: {e}")
    print("Please ensure all required helper scripts are present in the SCRIPTS_PATH.")
    sys.exit(1)
    

# %% NEW PLOTTING FUNCTION

def plot_prediction_probabilities(metadata, output_dir, testn, max_sequences=20):
    """
    Generates a FacetGrid of bar plots visualizing the prediction probabilities for each sequence.
    """
    print("\n Optional Plot showing ")
    print("----------------------------\n")
    
    # Convert metadata dictionary to DataFrame for easier grouping and plotting
    df = pd.DataFrame(metadata)
    
    # Ensure 'sequence' exists (assuming this is the key from GRAPH_ExtractEmbeddings)
    if 'sequences' not in df.columns:
        print("Error: 'sequences' not found in metadata. Cannot plot.")
        return

    # Sample sequences if there are too many
    unique_sequences = df['sequences'].unique()
    if len(unique_sequences) > max_sequences:
        print(f"Sampling {max_sequences} sequences out of {len(unique_sequences)} for visualization.")
        # Use a fixed seed for reproducible sampling
        sampled_sequences = pd.Series(unique_sequences).sample(n=max_sequences, random_state=42).tolist()
        df_plot = df[df['sequences'].isin(sampled_sequences)]
    else:
        df_plot = df

    # Set visual style
    sns.set_style("whitegrid")
    
    # Create a FacetGrid: one row per sequence. 
    # aspect and height control the dimensions of each facet.
    g = sns.FacetGrid(df_plot, row="sequences", aspect=6, height=2.0, sharex=False, sharey=True)
    
    # Define the plotting function to map onto the grid
    def plot_sequence_bars(data, **kwargs):
        # Ensure data is sorted by position if 'Position' column exists
        position_key = 'Position'
        if 'PositionInGraph' in data.columns:
             position_key = 'PositionInGraph'

        if position_key in data.columns:
            data = data.sort_values(by=position_key)
            x_axis = data[position_key]
        else:
            # Fallback to index if position key is missing
            data = data.reset_index(drop=True)
            x_axis = data.index

        probabilities = data['PredictionProbability']
        
        # Create colors based on probability (using a coolwarm colormap: Blue=Low, Red=High)
        colors = plt.cm.coolwarm(probabilities)

        ax = plt.gca()
        # width=1.0 ensures bars touch if positions are contiguous
        ax.bar(x_axis, probabilities, color=colors, width=1.0)
        
        # Add a line at 0.5 threshold for reference
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)


    g.map_dataframe(plot_sequence_bars)
    
    # Formatting the grid
    g.set_xlabels("Position")
    g.set_ylabels("Prob (Functional)")
    
    # Add a main title
    g.fig.suptitle("Predicted Functionality Probabilities", y=1.02, fontsize=16)
    
    # Save the plot
    PLOT_PATH = os.path.join(output_dir, f'{testn}_Probability_Plots.pdf')
    try:
        g.savefig(PLOT_PATH, bbox_inches='tight', dpi=300)
        print(f"Plots saved to {PLOT_PATH}")
    except Exception as e:
        print(f"Error saving plots: {e}")
        
# %% HELPER FUNCTIONS

def get_model_paths(model_type, base_dir):
    """Returns the paths for the GNN model, classifier, and scaler based on the model type.
    
    Args:
        model_type (str): 'XGB', 'RF', or 'TM' (Two Models using both)
        base_dir (str): Base directory containing the model files
        
    Returns:
        dict: Paths to model files. For TM, returns nested dict with 'XGB' and 'RF' keys.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Model base directory not found: {base_dir}")
    
    # Standardize model type input
    model_type = model_type.upper()
    
    if model_type in ["XGB", "RF"]:
        # Single model - assumes standard naming convention
        paths = {
            "GNN": os.path.join(base_dir, f"best_model_{model_type}.pth"),
            "Classifier": os.path.join(base_dir, f"{model_type}_classifier.joblib"),
            "Scaler": os.path.join(base_dir, f"scaler_{model_type}.pkl")
        }
        
        # Validate paths exist
        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Error: {name} file not found at {path} for model type {model_type}.")
                
        return paths
        
    elif model_type == "TM":
        # Team Model - load both XGB and RF models
        paths = {
            "GNN": os.path.join(base_dir, "best_model_TwoModels.pth"),
            "Classifier1": os.path.join(base_dir, "TwoModels_RF_classifier.joblib"),
            "Classifier2": os.path.join(base_dir, "TwoModels_XGB_classifier.joblib"),
            "Scaler": os.path.join(base_dir, "scaler_TwoModels.pkl"),
            "MetaModel": os.path.join(base_dir, "MetaLearner_LR_TwoModels.joblib")
        }
        
        # Validate all paths exist
        for component_name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Error: {component_name} file not found at {path} for TM setup.")
        
        return paths
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'XGB', 'RF', or 'TM'.")
# %% MAIN EXECUTION FUNCTION

def main(input_file, output_dir, model_type, generate_plot):
    
    # %% SETUP ENVIRONMENT AND DIRECTORIES
    print("\n=== Setting up environment ===")
    print("----------------------------\n")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    OUT_DIR = output_dir
    
    # Setup environment configuration
    SEED, DEVICE, WORKERS = setup_environment(seed=42)
    
    # Define a unique identifier for this run based on the input filename and model type
    TESTN = f"PRED_{os.path.splitext(os.path.basename(input_file))[0]}"

    print(f"✓ Input File: {input_file}")
    print(f"✓ Output Dir: {OUT_DIR}")
    print(f"✓ Model Type: {model_type}")
    print(f"✓ Configuration loaded (Device: {DEVICE}, Workers: {WORKERS})")
    print("\n----------------------------\n")
    
    # --- Determine Model Paths ---
    try:
        ModelPaths = get_model_paths(model_type, MODEL_BASE_DIR)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error configuring models: {e}")
        sys.exit(1)

    # %% LOADING NEW DATA
    print("\n=== LOADING NEW DATA FOR PREDICTION ===")
    print("----------------------------\n")
    start_time = time.perf_counter()

    # Define expected column names for the input coverage file
    colnames = ["sequence", "position", "coverage", "Prime3", "Prime5"]
 
    # Load the data.
    df_full, df, warning = load_and_prepare_data(input_file, sep=',', engine='python')
    
    df.columns = colnames

    if df is None:
        print("Error: Data loading failed. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from new data file.")
    # Print time
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n ✓ Completed in {duration:.2f} seconds.")

    print("\n----------------------------\n")
    gc.collect()

# %% MODEL LOADING
    print("\n=== MODEL LOADING FOR PREDICTION ===")
    print("----------------------------\n")
    start_time = time.perf_counter()
    
    # Embedding model configuration (GNN) - Must match the training configuration
    MODEL_TYPE = model_type.upper()
    ModelBlueprint = partial(create_graphsage_model, model_type="transformer_style")
    # 3 base features + 12 Generated Statistical features + 8 Positional Encoding features (sin and cos)
    INPUT_CHANNELS = 23  
    HIDDEN_CHANNELS = 64 
    
    # Load GNN Model
    try:
        print(f"Loading GNN weights from: {os.path.basename(ModelPaths['GNN'])}")
        model = load_prediction_model(
            model_class=ModelBlueprint,
            in_channels=INPUT_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            model_path=ModelPaths['GNN'],
            device=DEVICE
        )
    except Exception as e:
        print(f"Error loading GNN model: {e}")
        sys.exit(1)
    
    # Load Classifier(s) based on model type
    if MODEL_TYPE == "TM":
        # Two Models - load both classifiers
        try:
            print(f"Loading RF Classifier from: {os.path.basename(ModelPaths['Classifier1'])}")
            classifier_rf = load_classifier(ModelPaths['Classifier1'])
            
            print(f"Loading XGB Classifier from: {os.path.basename(ModelPaths['Classifier2'])}")
            classifier_xgb = load_classifier(ModelPaths['Classifier2'])
            
            # Store both classifiers for later use
            classifiers = {
                'RF': classifier_rf,
                'XGB': classifier_xgb
            }
            print("✓ Both classifiers loaded successfully for Two Models setup")
            
        except Exception as e:
            print(f"Error loading classifiers for TM: {e}")
            sys.exit(1)
    else:
        # Single model - load one classifier
        try:
            print(f"Loading {MODEL_TYPE} Classifier from: {os.path.basename(ModelPaths['Classifier'])}")
            classifier = load_classifier(ModelPaths['Classifier'])
        except Exception as e:
            print(f"Error loading {MODEL_TYPE} Classifier: {e}")
            sys.exit(1)
            
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n ✓ Model loading completed in {duration:.2f} seconds.") 
    print("\n----------------------------\n")
    gc.collect()

    # %% EXTRA FEATURES

    print("\n=== EXTRA FEATURES COMPUTATION ===")
    print("----------------------------\n")
    start_time = time.perf_counter()

    df_combined_features, feature_names = create_extra_features(df)

    # Print time
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n ✓ Completed in {duration:.2f} seconds.")

    print("\n----------------------------\n")
    gc.collect()

    # %% SCALING 

    print("\n=== SCALING OF FEATURES ===")
    print("----------------------------\n")
    start_time = time.perf_counter()

    try:
        # Load the pre-fitted scaler
        print(f"Loading Scaler from: {os.path.basename(ModelPaths['Scaler'])}")
        fitted_scaler = load_scaler(ModelPaths['Scaler'])

        # Use the loaded scaler's .transform() method on the new data's features
        print(f"Transforming {len(feature_names)} features...")
        
        # Create a copy and apply transformation
        df_scaled_features = df_combined_features.copy()
        df_scaled_features[feature_names] = fitted_scaler.transform(
            df_combined_features[feature_names]
        )
        print("✓ Scaling transform complete.")

    except Exception as e:
        print(f"Error during scaling: {e}")
        sys.exit(1)
        
    # Print time
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n ✓ Completed in {duration:.2f} seconds.")

    print("\n----------------------------\n")
    del df_combined_features
    gc.collect()

    # %% GENERATE GRAPHS
    print("\n=== GENERATE GRAPHS (FOR PREDICTION) ===")
    print("----------------------------\n")
    start_time = time.perf_counter()

    # Call the function that does NOT require a 'label' column.
    graph_data_list = create_pyg_graphs_for_prediction(
        df_processed=df_scaled_features,
        feature_cols=feature_names,
        max_len=150 # Ensure this matches the training configuration
    )

    print(f"\nTotal graphs created: {len(graph_data_list)}")

    # Print time
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n ✓ Completed in {duration:.2f} seconds.")

    print("\n----------------------------\n")
    del df_scaled_features
    gc.collect()


    # %% GENERATE EMBEDDINGS AND METADATA
    print("\n=== GENERATING NODE EMBEDDINGS & METADATA ===")
    print("----------------------------\n")
    start_time = time.perf_counter()

    # Use a DataLoader for efficient processing
    eval_loader = DataLoader(graph_data_list, batch_size=128, shuffle=False, num_workers=WORKERS)

    # --- Extract Embeddings (Fast Loop) ---
    print("Generating node embeddings...")
    all_embeddings_list = []
    # Use torch.inference_mode() for optimized inference (no gradient tracking)
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Generating Embeddings"):
            batch = batch.to(DEVICE)
            
            # Generate embeddings
            embeddings = model(batch.x, batch.edge_index, batch.batch)
            
            # Collect the tensor batches, move to CPU
            all_embeddings_list.append(embeddings.cpu())

    # Concatenate all embedding batches into a single large tensor, then to NumPy
    node_embeddings = torch.cat(all_embeddings_list, dim=0).numpy()

    # Extract metadata (Sequence Name, Position)
    print("Extracting metadata...")
    metadata = extract_prediction_metadata(graph_data_list)

    # The feature matrix 'X' is our embeddings.
    X = node_embeddings
    print(f"\nGenerated {X.shape[0]} node embeddings of dimension {X.shape[1]}.")

    # Print time
    end_time = time.perf_counter()
    duration = end_time - start_time

    print(f"\n ✓ Completed in {duration:.2f} seconds.")
    print("\n----------------------------\n")
    # Clean up memory
    del eval_loader, graph_data_list, model, all_embeddings_list
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



    # %% CLASSIFIER PREDICTION
    print("\n=== CLASSIFIER PREDICTION ===")
    print("------------------------------------------\n")
    start_time = time.perf_counter()
    
    if MODEL_TYPE == "TM":
        # Two Models - get predictions from both classifiers for meta-model
        print(f"Running Two Models prediction on {len(X)} nodes.")
        print("Getting predictions from RF classifier...")
        rf_pred = classifiers['RF'].predict(X)
        rf_proba = classifiers['RF'].predict_proba(X)[:, 1]
        
        print("Getting predictions from XGB classifier...")
        xgb_pred = classifiers['XGB'].predict(X)
        xgb_proba = classifiers['XGB'].predict_proba(X)[:, 1]
        
        # Create feature matrix for meta-model
        # Using probabilities as features for the meta-model
        meta_features = np.column_stack([rf_proba, xgb_proba])
        
        print("Applying meta-model on classifier outputs...")
        #Load and apply your meta-model here
        meta_model = load_classifier(ModelPaths['MetaModel'])
        y_pred = meta_model.predict(meta_features)
        y_proba = meta_model.predict_proba(meta_features)[:, 1]
        
        print("✓ Meta-model (LR) prediction complete.")
        
        # Optional: Store individual model predictions for analysis
        individual_predictions = {
            'RF': {'pred': rf_pred, 'proba': rf_proba},
            'XGB': {'pred': xgb_pred, 'proba': xgb_proba}
        }
        
    else:
        # Single model prediction (original behavior)
        print(f"Running {MODEL_TYPE} classifier prediction on {len(X)} nodes.")
        y_pred = classifier.predict(X)
        y_proba = classifier.predict_proba(X)[:, 1]
        print("✓ Prediction complete.")
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n ✓ Completed in {duration:.2f} seconds.")
    print("\n----------------------------\n")
    
    # Cleanup
    if MODEL_TYPE == "TM":
        del X, classifiers, rf_pred, rf_proba, xgb_pred, xgb_proba, meta_features
    else:
        del X, classifier
    gc.collect()
    # %% SAVE PREDICTIONS TABLE
    print("\n=== SAVING PREDICTION RESULTS ===")
    print("----------------------------\n")
    start_time = time.perf_counter()
    
    if MODEL_TYPE == "TM":
        # Two Models - save main predictions plus individual model results
        RESULTS_TABLE_PATH = os.path.join(OUT_DIR, f'{TESTN}_Predictions_TM.csv')
        
        # Add main meta-model predictions
        metadata['PredictedLabel'] = y_pred
        metadata['PredictionProbability'] = y_proba
        
        # Add individual model predictions for analysis
        metadata['RF_PredictedLabel'] = individual_predictions['RF']['pred']
        metadata['RF_PredictionProbability'] = individual_predictions['RF']['proba']
        metadata['XGB_PredictedLabel'] = individual_predictions['XGB']['pred']
        metadata['XGB_PredictionProbability'] = individual_predictions['XGB']['proba']
        
        print("Saving Two Models predictions with individual model results...")
        save_multipredictions_to_csv(
            metadata=metadata,
            save_path=RESULTS_TABLE_PATH
        )
        
        # Save a summary table with just the main predictions
        SUMMARY_TABLE_PATH = os.path.join(OUT_DIR, f'{TESTN}_Predictions_TM_Summary.csv')
        summary_metadata = {k: v for k, v in metadata.items() 
                            if not any(prefix in k for prefix in ['RF_', 'XGB_'])}
        save_multipredictions_to_csv(
            metadata=summary_metadata,
            save_path=SUMMARY_TABLE_PATH
        )
        print(f"✓ Detailed results saved to: {os.path.basename(RESULTS_TABLE_PATH)}")
        print(f"✓ Summary results saved to: {os.path.basename(SUMMARY_TABLE_PATH)}")
        
    else:
        # Single model - only one model
        RESULTS_TABLE_PATH = os.path.join(OUT_DIR, f'{TESTN}_Predictions_{MODEL_TYPE}_Summary.csv')
        
        # Add the predictions to the metadata dictionary
        metadata['PredictedLabel'] = y_pred
        metadata['PredictionProbability'] = y_proba
        
        print(f"Saving {MODEL_TYPE} predictions...")
        save_multipredictions_to_csv(
            metadata=metadata,
            save_path=RESULTS_TABLE_PATH
        )
        print(f"✓ Results saved to: {os.path.basename(RESULTS_TABLE_PATH)}")
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n ✓ Completed in {duration:.2f} seconds.")
    print("\n----------------------------\n")

    # %% GENERATE PLOTS (Optional)
    if generate_plot:
        plot_prediction_probabilities(metadata, OUT_DIR, TESTN)

    # %% CLUSTERING AND SAVE    
    print("\n=== CLUSTERING PREDICTIONS AND GENERATING SAF FILE ===")
    print("----------------------------\n")
    start_time = time.perf_counter()
    RESULTS_CLUSTERS_PATH = os.path.join(OUT_DIR, f'{TESTN}_ClustAnnotation.saf')
    predictions_df = pd.read_csv(RESULTS_TABLE_PATH)
    found_clusters = find_label1_clusters(
    df=predictions_df,
    min_cluster_size=15,
    tolerance=2 # Allow one '0' within a cluster
    )
    if found_clusters:
        output_filename = RESULTS_CLUSTERS_PATH
        create_saf_output(found_clusters, output_filename)
        print(f"Results saved to {output_filename}")
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n ✓ Completed in {duration:.2f} seconds.")
    print("\n----------------------------\n")    
    print("\n=== PREDICTION PIPELINE FINISHED ===")

# %% COMMAND LINE INTERFACE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GNN prediction pipeline on new coverage data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-I", "--input", required=True, 
                        help="Path to the input coverage file (TSV). Format: Seq, Pos, Cov, P3, P5 (no header).")
    
    parser.add_argument("-O", "--output_dir", required=True, 
                        help="Directory where results (predictions, plots) will be saved.")
    
    parser.add_argument("-M", "--model_type", required=True, choices=['XGB', 'RF', 'TM'],
                        help="The downstream classifier model to use (XGB, RF or combination of the two (TM)).")
    
    # action='store_true' means the presence of the flag sets the value to True.
    parser.add_argument("-P", "--plot", action='store_true', 
                        help="If set, generate visualization plots of the prediction probabilities.")
    
    # Handle case where no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print(f"\nNote: Models are expected to be located in: {MODEL_BASE_DIR}")
        print("\nExample Usage:")
        print("python PIPELINE_Prediction.py -I /path/to/input.csv -O /path/to/output -M XGB -P")
        sys.exit(1)

    args = parser.parse_args()
    
    main(args.input, args.output_dir, args.model_type, args.plot)
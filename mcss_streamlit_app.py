#!/usr/bin/env python3
"""MCSS Combined Script with Streamlit Interface

A Streamlit version of the Monte Carlo Subsampling for Clustering Replicability (MCSS) script
that supports direct file upload without requiring Google Drive integration.

Features:
- Direct file upload via Streamlit interface
- Supports both .npy and .csv datasets
- Real-time streaming progress updates
- Deterministic clustering results
- Combined KMeans and Agglomerative Clustering support
- Interactive metrics exploration
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from typing import Union, List, Dict, Tuple, Optional
import shutil
from collections import deque
import time
import json
from hashlib import sha256
import tempfile
import zipfile
import threading
import queue

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
except Exception as _e:
    raise RuntimeError("scikit-learn is required. Please install scikit-learn before running.") from _e

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except Exception:
    _STREAMLIT_AVAILABLE = False

try:
    import plotly.express as px
    _PLOTLY_AVAILABLE = True
except Exception:
    _PLOTLY_AVAILABLE = False


######################
# Configuration constants (defaults)
######################

# Use current working directory as default base
_DEFAULT_BASE_DEST = os.getcwd()

DEFAULT_CONFIG = {
    'B': 200,  # Number of Monte Carlo iterations
    'PERCENT_SUBSAMPLE': 0.8,  # Proportion of data for training subsample
    'OUTPUT_DIR': os.path.join(_DEFAULT_BASE_DEST, "MCSS_Streamlit_Output"),  # Base directory for all output runs
    'DATA_PATH': os.path.join(_DEFAULT_BASE_DEST, "temp_datasets_streamlit"),  # Path to temporary uploaded files
    'RANDOM_SEED': 123,
    'CLUSTER_RANGE': list(range(2, 6)),  # Range or list of cluster numbers (k) to evaluate
    'TEMP_DIR': "temp_results_streamlit",  # Base temporary directory for intermediate files
    'SELECTED_DATASETS': [],  # Will be populated from uploaded files
    'METHOD': 'both',  # Choose 'kmeans', 'agglomerative', or 'both'
    'LINKAGES': ["single", "ward"],  # For agglomerative clustering
}


######################
# Streamlit Session State Initialization
######################

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'run_status' not in st.session_state:
        st.session_state.run_status = {
            'total_configs': 0, 'current_index': 0, 'phase': '', 'start_time': 0.0,
            'output_folders': [], 'last_error': '', 'is_running': False,
        }
    if 'cancel_requested' not in st.session_state:
        st.session_state.cancel_requested = False
    if 'uploaded_files_info' not in st.session_state:
        st.session_state.uploaded_files_info = []


######################
# Utilities
######################

def log_progress(message: str, detail_level: int = 1) -> None:
    """Prints a progress message with a timestamp and stores for UI."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    indent = "  " * (detail_level - 1)
    formatted = f"[{timestamp}] {indent}{message}"
    print(formatted)
    
    # Store in session state for Streamlit (only if running in Streamlit context)
    try:
        if _STREAMLIT_AVAILABLE and 'log_messages' in st.session_state:
            st.session_state.log_messages.append(formatted)
    except (NameError, AttributeError):
        # Not in Streamlit context, just continue
        pass


def get_logs_for_ui() -> str:
    """Returns accumulated logs as a string."""
    try:
        if _STREAMLIT_AVAILABLE and hasattr(st, 'session_state') and 'log_messages' in st.session_state:
            return "\n".join(st.session_state.log_messages)
    except (NameError, AttributeError):
        pass
    return ""


def reset_run_status(total_configs: int) -> None:
    """Reset run status for a new run."""
    try:
        if _STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
            st.session_state.run_status.update({
                'total_configs': total_configs,
                'current_index': 0,
                'phase': 'initializing',
                'start_time': time.time(),
                'output_folders': [],
                'is_running': True,
                'last_error': '',
            })
            st.session_state.log_messages = [] # Clear previous logs
            st.session_state.cancel_requested = False # Reset cancel flag
    except (NameError, AttributeError):
        pass


def overall_progress() -> float:
    """Calculate overall progress percentage."""
    try:
        if _STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
            total = st.session_state.run_status['total_configs']
            idx = st.session_state.run_status['current_index']
            if total <= 0:
                return 0.0
            return max(0.0, min(1.0, idx / total))
    except (NameError, AttributeError, KeyError):
        pass
    return 0.0


def eta_text() -> str:
    """Calculate estimated time to completion."""
    try:
        if _STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
            total = st.session_state.run_status['total_configs']
            idx = st.session_state.run_status['current_index']
            if total == 0 or idx == 0:
                return "ETA: --"
            elapsed = time.time() - st.session_state.run_status['start_time']
            avg = elapsed / idx
            remaining = max(0, total - idx)
            eta_sec = int(remaining * avg)
            return f"ETA: ~{eta_sec//60}m {eta_sec%60}s"
    except (NameError, AttributeError, KeyError):
        pass
    return "ETA: --"


def compute_config_hash(cfg: Dict[str, object]) -> str:
    """Compute a hash for the configuration."""
    minimal = {
        'B': cfg.get('B'),
        'PERCENT_SUBSAMPLE': cfg.get('PERCENT_SUBSAMPLE'),
        'RANDOM_SEED': cfg.get('RANDOM_SEED'),
        'METHOD': cfg.get('METHOD'),
        'LINKAGES': cfg.get('LINKAGES'),
        'CLUSTER_RANGE': cfg.get('CLUSTER_RANGE'),
        'SELECTED_DATASETS': cfg.get('SELECTED_DATASETS'),
    }
    s = json.dumps(minimal, sort_keys=True, default=str)
    return sha256(s.encode('utf-8')).hexdigest()[:12]


def determinism_badge_md() -> str:
    """Generate determinism information markdown."""
    seed = DEFAULT_CONFIG.get('RANDOM_SEED')
    caps = {
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', ''),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS', ''),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS', ''),
        'VECLIB_MAXIMUM_THREADS': os.environ.get('VECLIB_MAXIMUM_THREADS', ''),
        'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS', ''),
    }
    cfg_hash = compute_config_hash(DEFAULT_CONFIG)
    caps_str = ", ".join(f"{k}={v or '-'}" for k, v in caps.items())
    return f"**Determinism**: seed={seed}, {caps_str} • cfg-hash={cfg_hash}"


def cancel_run() -> str:
    """Cancel the current run."""
    st.session_state.cancel_requested = True
    log_progress("[CANCEL] User requested cancellation.")
    return "Cancel requested. Finishing current step..."


def create_results_zip(output_folders: List[str]) -> Optional[bytes]:
    """Create a zip file containing all result folders."""
    if not output_folders:
        return None
    
    try:
        # Create a temporary zip file
        zip_path = os.path.join(tempfile.gettempdir(), f"mcss_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for folder_path in output_folders:
                if os.path.exists(folder_path):
                    folder_name = os.path.basename(folder_path)
                    # Add all files in the folder to the zip
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Create archive path relative to the folder
                            arcname = os.path.join(folder_name, os.path.relpath(file_path, folder_path))
                            zipf.write(file_path, arcname)
                            
        with open(zip_path, "rb") as f:
            return f.read()
        
    except Exception as e:
        log_progress(f"Error creating zip file: {e}")
        return None


def set_deterministic_mode(seed_value: int) -> None:
    """Enforce stronger run-to-run determinism across libraries and threads."""
    try:
        os.environ.setdefault("PYTHONHASHSEED", str(seed_value))
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    except Exception:
        pass

    random.seed(seed_value)
    np.random.seed(seed_value)


def process_streamlit_file(uploaded_file) -> Tuple[str, str]:
    """Process uploaded file and return dataset info."""
    if uploaded_file is None:
        raise ValueError("No file uploaded")
    
    # Get the original filename
    original_name = uploaded_file.name
    
    # Create a safe filename
    safe_name = os.path.basename(original_name)
    dataset_name = os.path.splitext(safe_name)[0]
    
    # Ensure data directory exists
    data_dir = DEFAULT_CONFIG['DATA_PATH']
    os.makedirs(data_dir, exist_ok=True)
    
    # Copy uploaded file to our data directory
    dest_path = os.path.join(data_dir, safe_name)
    
    # Write the uploaded file content
    with open(dest_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    log_progress(f"Uploaded file saved as: {dest_path}")
    log_progress(f"Dataset name: {dataset_name}")
    
    return dataset_name, safe_name


# Import all the clustering functions from the original script
# (These functions remain the same as they are pure computational functions)

def load_raw_data(dataset_info_list: List[Dict[str, str]], data_path_str: str) -> Dict[str, pd.DataFrame]:
    """Loads raw data for specified datasets from either .npy or .csv files."""
    datasets: Dict[str, pd.DataFrame] = {}
    for info in dataset_info_list:
        name = info['name']
        filename = info['filename']
        file_path = os.path.join(data_path_str, filename)

        if not os.path.exists(file_path):
            log_progress(f"Warning: Data file not found for dataset '{name}': {file_path}", detail_level=2)
            continue

        try:
            if filename.endswith('.npy'):
                # Assumes the .npy file contains a dict with an 'all' key holding a DataFrame.
                loaded_dict = np.load(file_path, allow_pickle=True).item()
                if isinstance(loaded_dict, dict) and 'all' in loaded_dict:
                    datasets[name] = loaded_dict['all']
                else:
                    # If it's just a numpy array, convert to DataFrame
                    datasets[name] = pd.DataFrame(loaded_dict)
                log_progress(f"Successfully loaded .npy dataset: {name}")
            elif filename.endswith('.csv'):
                # Load raw CSV as-is; header handling is deferred
                datasets[name] = pd.read_csv(file_path, header=None, low_memory=False)
                log_progress(f"Successfully loaded .csv dataset: {name}")
            else:
                log_progress(f"Warning: Unsupported file type for {filename}. Skipping.", detail_level=2)
        except Exception as e:
            log_progress(f"Error loading dataset '{name}' from {file_path}: {e}", detail_level=2)

    if not datasets:
        raise ValueError("No valid datasets could be loaded from the provided list.")

    return datasets


def prepare_samples_array(samples_df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Convert a DataFrame to a numeric array for clustering."""
    if not isinstance(samples_df, pd.DataFrame):
        return samples_df  # Already numeric array

    # Header sniffing: check if the first row (excluding first column) looks non-numeric
    if not samples_df.empty and samples_df.shape[1] > 1:
        first_row_values = samples_df.iloc[0, 1:]
        non_numeric_count = pd.to_numeric(first_row_values, errors='coerce').isna().sum()
        if non_numeric_count > len(first_row_values) / 2:
            log_progress("Header-like first row detected. Skipping it.", detail_level=3)
            samples_df = samples_df.iloc[1:].reset_index(drop=True)

    # If the first column is non-numeric, assume gene identifiers
    df_to_process = samples_df
    if not df_to_process.empty and not np.issubdtype(df_to_process.iloc[:, 0].dtype, np.number):
        log_progress("First column is non-numeric, skipping as potential gene identifiers.", detail_level=3)
        df_to_process = df_to_process.iloc[:, 1:]

    # Retain only numeric columns
    numeric_df = df_to_process.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < df_to_process.shape[1]:
        dropped_cols = df_to_process.shape[1] - numeric_df.shape[1]
        log_progress(f"Warning: Dropped {dropped_cols} non-numeric sample columns.", detail_level=2)

    # Transpose to Samples-by-Features
    samples = numeric_df.values.T

    if not np.issubdtype(samples.dtype, np.number):
        raise ValueError("Could not create a numeric array for clustering. Check for non-numeric values in your data.")

    return samples


# Copy all the clustering functions from the original file
# (monte_carlo_subsampling, load_iteration_data, calculate_and_save_sorted_centroids_kmeans, etc.)
# For brevity, I'll include just the key ones and indicate where others should be copied

def monte_carlo_subsampling(samples_array: np.ndarray,
                            num_samples: int,
                            num_iterations: int,
                            subsample_size_train: int,
                            base_save_folder_str: str,
                            iter_prefix: str = "") -> Tuple[str, str]:
    """Monte Carlo subsampling: create train/test splits and store .npy arrays + index lists."""
    log_progress("Starting Monte Carlo subsampling")
    subsamples_data_folder = os.path.join(base_save_folder_str, f'{iter_prefix}subsamples_data')
    indices_folder = os.path.join(base_save_folder_str, f'{iter_prefix}indices')
    
    # Create directories and handle permission issues
    import stat
    try:
        os.makedirs(subsamples_data_folder, exist_ok=True)
        os.makedirs(indices_folder, exist_ok=True)
        
        # If directories exist, try to clean up any existing files with permission issues
        if os.path.exists(subsamples_data_folder):
            for existing_file in os.listdir(subsamples_data_folder):
                file_path = os.path.join(subsamples_data_folder, existing_file)
                try:
                    if os.path.isfile(file_path):
                        os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)
                        os.remove(file_path)
                except (OSError, PermissionError) as e:
                    log_progress(f"Warning: Could not remove existing file {file_path}: {e}", detail_level=3)
                    
    except OSError as e:
        log_progress(f"Warning: Directory creation issue: {e}", detail_level=2)

    all_subset_indices_list: List[np.ndarray] = []
    all_complement_indices_list: List[np.ndarray] = []

    for B_idx_iter in range(num_iterations):
        if B_idx_iter % 20 == 0:
            log_progress(f"Subsampling iteration {B_idx_iter+1}/{num_iterations}", detail_level=2)

        current_indices = random.sample(range(num_samples), num_samples)
        subset_indices_arr = np.array(current_indices[:subsample_size_train], dtype=int)
        complement_indices_arr = np.array(current_indices[subsample_size_train:], dtype=int)

        all_subset_indices_list.append(subset_indices_arr)
        all_complement_indices_list.append(complement_indices_arr)

        subset_data_arr = samples_array[subset_indices_arr]
        complement_data_arr = samples_array[complement_indices_arr]

        # Save with error handling
        try:
            subset_file_path = os.path.join(subsamples_data_folder, f'subset_data_{B_idx_iter}.npy')
            complement_file_path = os.path.join(subsamples_data_folder, f'complement_data_{B_idx_iter}.npy')
            
            np.save(subset_file_path, subset_data_arr)
            np.save(complement_file_path, complement_data_arr)
        except PermissionError as e:
            log_progress(f"Permission error saving files for iteration {B_idx_iter}: {e}", detail_level=2)
            # Try to fix permissions on the directory
            try:
                import stat
                os.chmod(subsamples_data_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                np.save(subset_file_path, subset_data_arr)
                np.save(complement_file_path, complement_data_arr)
                log_progress(f"Successfully saved files after permission fix for iteration {B_idx_iter}", detail_level=3)
            except Exception as retry_e:
                log_progress(f"Failed to save files even after permission fix for iteration {B_idx_iter}: {retry_e}", detail_level=2)
                raise

    np.save(os.path.join(indices_folder, 'all_train_indices.npy'), np.array(all_subset_indices_list, dtype=object))
    np.save(os.path.join(indices_folder, 'all_test_indices.npy'), np.array(all_complement_indices_list, dtype=object))

    log_progress("Completed Monte Carlo subsampling")
    return subsamples_data_folder, indices_folder


def load_iteration_data(B_idx_iter: int, subsamples_data_folder_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load iteration data from saved files."""
    subset_data_arr = np.load(os.path.join(subsamples_data_folder_str, f'subset_data_{B_idx_iter}.npy'))
    complement_data_arr = np.load(os.path.join(subsamples_data_folder_str, f'complement_data_{B_idx_iter}.npy'))
    return subset_data_arr, complement_data_arr


# ----------------------- K-means specific -----------------------

def calculate_and_save_sorted_centroids_kmeans(samples_array: np.ndarray,
                                                num_clusters: int,
                                                method_str_val: str,
                                                base_temp_run_folder_str: str) -> np.ndarray:
    """Calculate and save sorted centroids for K-means."""
    log_progress("Calculating global centroids using K-means clustering on full dataset")
    clustering_model = KMeans(
        n_clusters=num_clusters,
        random_state=0,
        n_init="auto"
    ).fit(samples_array)
    centroid_values_arr = clustering_model.cluster_centers_

    log_progress("Sorting global centroids by L2 norm", detail_level=2)
    l2_norms_arr = [np.linalg.norm(row) for row in centroid_values_arr]
    rows_with_norms_list = list(zip(centroid_values_arr, l2_norms_arr))
    centroid_values_sorted_list = [row for row, norm_val in sorted(rows_with_norms_list, key=lambda x: x[1])]
    centroid_values_sorted_arr = np.array(centroid_values_sorted_list)

    method_specific_folder = os.path.join(base_temp_run_folder_str, f'{method_str_val}_{num_clusters}_clusters')
    os.makedirs(method_specific_folder, exist_ok=True)

    np.save(os.path.join(method_specific_folder, 'cluster_centers.npy'), centroid_values_arr)
    pd.DataFrame(centroid_values_arr).to_csv(os.path.join(method_specific_folder, 'cluster_centers.csv'), index=False, header=False)

    np.save(os.path.join(method_specific_folder, 'centroid_values_sorted.npy'), centroid_values_sorted_arr)
    pd.DataFrame(centroid_values_sorted_arr).to_csv(os.path.join(method_specific_folder, 'centroid_values_sorted.csv'), index=False, header=False)

    log_progress("Completed global centroid calculation and sorting")
    return centroid_values_sorted_arr


def perform_kmeans_clustering_and_get_centroids(num_iterations: int,
                                                num_clusters: int,
                                                subsamples_data_folder_str: str,
                                                iter_cluster_results_save_path: str,
                                                vvw_files_save_path: str,
                                                method_str_val: str,
                                                complement_size: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """K-means MCSS: fit on training subset, predict test subset. Keep results in memory only."""
    log_progress(f"Starting K-means clustering (fitting on training data) for {num_iterations} iterations")

    predicted_labels_matrix = np.zeros((num_iterations, complement_size))
    iteration_centers_list: List[np.ndarray] = []

    for B_idx_iter in range(num_iterations):
        if B_idx_iter % 20 == 0:
            log_progress(f"K-means Iteration {B_idx_iter+1}/{num_iterations}", detail_level=2)

        subset_data_arr, complement_data_arr = load_iteration_data(B_idx_iter, subsamples_data_folder_str)

        try:
            log_progress(f"Fitting K-means on TRAINING data for iteration {B_idx_iter+1}", detail_level=3)
            kmeans_model = KMeans(
                n_clusters=num_clusters,
                random_state=0,
                n_init="auto"
            )
            kmeans_model.fit(subset_data_arr)
            predictions_arr = kmeans_model.predict(complement_data_arr)

            predicted_labels_matrix[B_idx_iter, :] = predictions_arr
            current_iter_centroids = kmeans_model.cluster_centers_
            iteration_centers_list.append(current_iter_centroids)

            log_progress("Processed iteration K-means results (training data fit, kept in memory)", detail_level=3)
        except Exception as e:
            log_progress(f"Error in K-means iteration {B_idx_iter} (training data fit): {str(e)}", detail_level=2)
            predicted_labels_matrix[B_idx_iter, :] = np.nan
            iteration_centers_list.append(np.full((num_clusters, complement_data_arr.shape[1]), np.nan))

    return predicted_labels_matrix, iteration_centers_list


def align_cluster_identities_kmeans(num_iterations: int,
                                    num_test_samples: int,
                                    num_clusters: int,
                                    global_sorted_centroids_arr: np.ndarray,
                                    unaligned_predictions_matrix: np.ndarray,
                                    iteration_centroids_list: List[np.ndarray],
                                    method_cluster_path_str: str,
                                    method_str_val: str,
                                    temp_save_folder_str: str) -> np.ndarray:
    """Align iteration labels to global centroids (K-means logic)."""
    aligned_predictions_matrix = np.zeros((num_iterations, num_test_samples))
    log_progress("Starting cluster identity alignment (K-means specific, from memory)")

    for B_idx_iter in range(num_iterations):
        if B_idx_iter % 20 == 0:
            log_progress(f"Aligning K-means iteration {B_idx_iter+1}/{num_iterations}", detail_level=2)

        iteration_centroids_arr = iteration_centroids_list[B_idx_iter]
        unaligned_preds_this_iter_arr = unaligned_predictions_matrix[B_idx_iter, :].flatten().astype(int)

        if np.isnan(iteration_centroids_arr).any():
            log_progress(f"Warning: Iteration centers contain NaNs, skipping alignment for iter {B_idx_iter}", detail_level=2)
            aligned_predictions_matrix[B_idx_iter, :] = np.nan
            continue

        dist_matrix = np.sqrt(np.sum((iteration_centroids_arr[:, np.newaxis, :] - global_sorted_centroids_arr[np.newaxis, :, :])**2, axis=2))

        mapping_from_iter_to_global = np.zeros(num_clusters, dtype=int)
        used_global_indices = np.zeros(num_clusters, dtype=bool)

        for i in range(num_clusters):
            available_distances = dist_matrix[i, ~used_global_indices]
            if len(available_distances) == 0:
                log_progress(f"Error in alignment mapping iter {B_idx_iter}, cluster {i}. Not enough global centroids.", detail_level=3)
                mapping_from_iter_to_global[i] = -1
                continue
            min_dist_idx_local = np.argmin(available_distances)
            original_global_idx = np.where(~used_global_indices)[0][min_dist_idx_local]
            mapping_from_iter_to_global[i] = original_global_idx
            used_global_indices[original_global_idx] = True

        transformed_preds_this_iter = np.zeros_like(unaligned_preds_this_iter_arr, dtype=float)
        for point_idx, original_cluster_label in enumerate(unaligned_preds_this_iter_arr):
            if 0 <= original_cluster_label < num_clusters and mapping_from_iter_to_global[original_cluster_label] != -1:
                transformed_preds_this_iter[point_idx] = mapping_from_iter_to_global[original_cluster_label]
            else:
                transformed_preds_this_iter[point_idx] = np.nan
                log_progress(f"Warning: Bad prediction label ({original_cluster_label}) or mapping for iter {B_idx_iter}, point {point_idx}", detail_level=3)

        aligned_predictions_matrix[B_idx_iter, :] = transformed_preds_this_iter

    log_progress("Completed K-means cluster identity alignment")
    return aligned_predictions_matrix


def generate_clam_matrix_kmeans(samples_array: np.ndarray,
                                all_test_indices_list_of_arrays: np.ndarray,
                                aligned_predictions_matrix: np.ndarray,
                                num_clusters: int,
                                method_cluster_path_str: str,
                                dataset_name: str,
                                method_str_val_param: str) -> np.ndarray:
    """Generate final CLAM matrix (summed across iterations). No per-iteration saves."""
    num_samples_total = samples_array.shape[0]
    final_clam_matrix = np.zeros((num_samples_total, num_clusters))

    for B_idx_iter in range(aligned_predictions_matrix.shape[0]):
        iter_clam_matrix = np.zeros((num_samples_total, num_clusters))
        current_iter_test_indices = all_test_indices_list_of_arrays[B_idx_iter]
        current_iter_aligned_preds = aligned_predictions_matrix[B_idx_iter, :]

        for i, sample_idx_val in enumerate(current_iter_test_indices):
            aligned_cluster_id = current_iter_aligned_preds[i]
            if not np.isnan(aligned_cluster_id):
                iter_clam_matrix[sample_idx_val, int(aligned_cluster_id)] += 1

        final_clam_matrix += iter_clam_matrix

    base_clam_filename = f'clam_{dataset_name}_k{num_clusters}_{method_str_val_param}'
    final_clam_csv_path = os.path.join(method_cluster_path_str, f'{base_clam_filename}.csv')
    final_clam_npy_path = os.path.join(method_cluster_path_str, 'clam_matrix.npy')

    pd.DataFrame(final_clam_matrix).to_csv(final_clam_csv_path, index=False, header=False)
    np.save(final_clam_npy_path, final_clam_matrix)
    log_progress(f"Final CLAM matrix CSV saved to {final_clam_csv_path}")
    log_progress(f"Final CLAM matrix NPY saved to {final_clam_npy_path}")

    return final_clam_matrix


# ----------------------- HC (agglomerative) specific -----------------------

def calculate_and_save_sorted_centroids_hc(samples_array: np.ndarray,
                                           num_clusters: int,
                                           method_str_val: str,
                                           linkage_val: str,
                                           base_temp_run_folder_str: str) -> np.ndarray:
    """Calculate and save sorted centroids for Agglomerative Clustering."""
    log_progress(f"Calculating global centroids using Agglomerative Clustering ({linkage_val}) on full dataset")

    clustering_model = AgglomerativeClustering(
        n_clusters=num_clusters,
        linkage=linkage_val
    ).fit(samples_array)

    centroids_list: List[np.ndarray] = []
    for i in range(num_clusters):
        cluster_points_arr = samples_array[clustering_model.labels_ == i]
        if cluster_points_arr.shape[0] > 0:
            centroid_arr = np.mean(cluster_points_arr, axis=0)
            centroids_list.append(centroid_arr)
        else:
            log_progress(f"Warning: Global cluster {i} is empty. Using mean of all samples as placeholder.", detail_level=2)
            centroids_list.append(np.mean(samples_array, axis=0))

    centroid_values_arr = np.array(centroids_list)

    log_progress("Sorting global centroids by L2 norm", detail_level=2)
    l2_norms_arr = [np.linalg.norm(row) for row in centroid_values_arr]
    rows_with_norms_list = list(zip(centroid_values_arr, l2_norms_arr))
    centroid_values_sorted_list = [row for row, norm_val in sorted(rows_with_norms_list, key=lambda x: x[1])]
    centroid_values_sorted_arr = np.array(centroid_values_sorted_list)

    method_specific_folder = os.path.join(base_temp_run_folder_str, f'{method_str_val}_{num_clusters}_clusters')
    os.makedirs(method_specific_folder, exist_ok=True)

    np.save(os.path.join(method_specific_folder, 'cluster_centers.npy'), centroid_values_arr)
    pd.DataFrame(centroid_values_arr).to_csv(os.path.join(method_specific_folder, 'cluster_centers.csv'), index=False, header=False)

    np.save(os.path.join(method_specific_folder, 'centroid_values_sorted.npy'), centroid_values_sorted_arr)
    pd.DataFrame(centroid_values_sorted_arr).to_csv(os.path.join(method_specific_folder, 'centroid_values_sorted.csv'), index=False, header=False)

    log_progress("Completed global centroid calculation and sorting")
    return centroid_values_sorted_arr


def perform_hc_clustering_and_get_centroids(num_iterations: int,
                                            num_clusters: int,
                                            linkage_val: str,
                                            subsamples_data_folder_str: str,
                                            iter_cluster_results_save_path: str,
                                            vvw_files_save_path: str,
                                            method_str_val: str,
                                            complement_size: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """HC MCSS: fit on test subset per the original HC logic. Keep results in memory only."""
    log_progress(f"Starting HC clustering (fitting on test data) for {num_iterations} iterations")

    predicted_labels_matrix = np.zeros((num_iterations, complement_size))
    iteration_centers_list: List[np.ndarray] = []

    for B_idx_iter in range(num_iterations):
        if B_idx_iter % 20 == 0:
            log_progress(f"HC Iteration {B_idx_iter+1}/{num_iterations}", detail_level=2)

        _subset_data_arr, complement_data_arr = load_iteration_data(B_idx_iter, subsamples_data_folder_str)

        try:
            log_progress(f"Fitting Agglomerative Clustering on TEST data for iteration {B_idx_iter+1}", detail_level=3)
            hc_model_test = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage=linkage_val
            )
            predictions_arr = hc_model_test.fit_predict(complement_data_arr)

            predicted_labels_matrix[B_idx_iter, :] = predictions_arr

            current_iter_centroids = np.zeros((num_clusters, complement_data_arr.shape[1]))
            test_data_labels = hc_model_test.labels_
            for cluster_idx_val in range(num_clusters):
                cluster_mask = (test_data_labels == cluster_idx_val)
                vvw_matrix = complement_data_arr[cluster_mask]
                if vvw_matrix.shape[0] > 0:
                    current_iter_centroids[cluster_idx_val] = np.mean(vvw_matrix, axis=0)
                else:
                    log_progress(f"Warning: Test data cluster {cluster_idx_val} empty in iter {B_idx_iter}. Using mean of test data as placeholder.", detail_level=3)
                    current_iter_centroids[cluster_idx_val] = np.mean(complement_data_arr, axis=0)

            iteration_centers_list.append(current_iter_centroids)
            log_progress("Processed iteration HC results (test data fit, kept in memory)", detail_level=3)
        except Exception as e:
            log_progress(f"Error in HC iteration {B_idx_iter} (test data fit): {str(e)}", detail_level=2)
            predicted_labels_matrix[B_idx_iter, :] = np.nan
            iteration_centers_list.append(np.full((num_clusters, complement_data_arr.shape[1]), np.nan))

    return predicted_labels_matrix, iteration_centers_list


def align_cluster_identities_hc(num_iterations: int,
                                num_test_samples: int,
                                num_clusters: int,
                                global_sorted_centroids_arr: np.ndarray,
                                unaligned_predictions_matrix: np.ndarray,
                                iteration_centroids_list: List[np.ndarray],
                                method_cluster_path_str: str,
                                method_str_val: str,
                                temp_save_folder_str: str) -> np.ndarray:
    """Align iteration labels to global centroids (HC logic)."""
    aligned_predictions_matrix = np.zeros((num_iterations, num_test_samples))
    log_progress("Starting cluster identity alignment (HC specific, from memory)")

    for B_idx_iter in range(num_iterations):
        if B_idx_iter % 20 == 0:
            log_progress(f"Aligning HC iteration {B_idx_iter+1}/{num_iterations}", detail_level=2)

        iteration_centroids_arr = iteration_centroids_list[B_idx_iter]
        unaligned_preds_this_iter_arr = unaligned_predictions_matrix[B_idx_iter, :].flatten().astype(int)

        if np.isnan(iteration_centroids_arr).any():
            log_progress(f"Warning: Iteration centers contain NaNs, skipping alignment for iter {B_idx_iter}", detail_level=2)
            aligned_predictions_matrix[B_idx_iter, :] = np.nan
            continue

        dist_matrix = np.sqrt(np.sum((iteration_centroids_arr[:, np.newaxis, :] - global_sorted_centroids_arr[np.newaxis, :, :])**2, axis=2))

        mapping_from_iter_to_global = np.zeros(num_clusters, dtype=int)
        used_global_indices = np.zeros(num_clusters, dtype=bool)

        for i in range(num_clusters):
            available_distances = dist_matrix[i, ~used_global_indices]
            if len(available_distances) == 0:
                log_progress(f"Error in alignment mapping iter {B_idx_iter}, cluster {i}. Not enough global centroids.", detail_level=3)
                mapping_from_iter_to_global[i] = -1
                continue
            min_dist_idx_local = np.argmin(available_distances)
            original_global_idx = np.where(~used_global_indices)[0][min_dist_idx_local]
            mapping_from_iter_to_global[i] = original_global_idx
            used_global_indices[original_global_idx] = True

        transformed_preds_this_iter = np.zeros_like(unaligned_preds_this_iter_arr, dtype=float)
        for point_idx, original_cluster_label in enumerate(unaligned_preds_this_iter_arr):
            if 0 <= original_cluster_label < num_clusters and mapping_from_iter_to_global[original_cluster_label] != -1:
                transformed_preds_this_iter[point_idx] = mapping_from_iter_to_global[original_cluster_label]
            else:
                transformed_preds_this_iter[point_idx] = np.nan
                log_progress(f"Warning: Bad prediction label ({original_cluster_label}) or mapping for iter {B_idx_iter}, point {point_idx}", detail_level=3)

        aligned_predictions_matrix[B_idx_iter, :] = transformed_preds_this_iter

    log_progress("Completed HC cluster identity alignment")
    return aligned_predictions_matrix


def generate_clam_matrix_hc(samples_array: np.ndarray,
                            all_test_indices_list_of_arrays: np.ndarray,
                            aligned_predictions_matrix: np.ndarray,
                            num_clusters: int,
                            method_cluster_path_str: str,
                            dataset_name: str,
                            method_str_val_param: str) -> np.ndarray:
    """Generate final CLAM matrix (summed across iterations) for HC."""
    num_samples_total = samples_array.shape[0]
    final_clam_matrix = np.zeros((num_samples_total, num_clusters))

    for B_idx_iter in range(aligned_predictions_matrix.shape[0]):
        iter_clam_matrix = np.zeros((num_samples_total, num_clusters))
        current_iter_test_indices = all_test_indices_list_of_arrays[B_idx_iter]
        current_iter_aligned_preds = aligned_predictions_matrix[B_idx_iter, :]

        for i, sample_idx_val in enumerate(current_iter_test_indices):
            aligned_cluster_id = current_iter_aligned_preds[i]
            if not np.isnan(aligned_cluster_id):
                iter_clam_matrix[sample_idx_val, int(aligned_cluster_id)] += 1

        final_clam_matrix += iter_clam_matrix

    base_clam_filename = f'clam_{dataset_name}_k{num_clusters}_{method_str_val_param}'
    final_clam_csv_path = os.path.join(method_cluster_path_str, f'{base_clam_filename}.csv')
    pd.DataFrame(final_clam_matrix).to_csv(final_clam_csv_path, index=False, header=False)
    log_progress(f"Final CLAM matrix CSV saved to {final_clam_csv_path}")

    return final_clam_matrix


def copy_results_to_destination(temp_folder: str, dest_folder: str, method_str: str, config_dict: Dict[str, Union[str, int]]) -> None:
    """Copy selected results from temp run folder to final destination."""
    if not os.path.exists(temp_folder):
        raise FileNotFoundError(f"Temporary folder not found: {temp_folder}")

    log_progress(f"Copying results from {temp_folder} to {dest_folder}")
    os.makedirs(dest_folder, exist_ok=True)

    method_cluster_dir_name = f'{method_str}_{config_dict["n_clusters"]}_clusters'

    # What to copy from temp to final destination
    items_to_copy = [
        ('config_info.yaml', '', False),
        ('samples_original.csv', '', False),
        ('aligned_matrix.npy', '', False),
        (f'{method_str}_predicted_testdata_80_20_aligned.csv', '', False),
        (method_cluster_dir_name, '', True),
    ]

    rsync_available = shutil.which("rsync") is not None

    for item_name, dest_subdir, is_dir_flag in items_to_copy:
        src_path = os.path.join(temp_folder, item_name)
        dest_path_final = os.path.join(dest_folder, dest_subdir, item_name) if dest_subdir else os.path.join(dest_folder, item_name)

        os.makedirs(os.path.dirname(dest_path_final), exist_ok=True)

        if os.path.exists(src_path):
            if is_dir_flag:
                if rsync_available:
                    cmd = f'rsync -av --progress "{src_path}/" "{dest_path_final}/"'
                    result = os.system(cmd)
                    if result == 0:
                        log_progress(f"Rsynced directory: '{item_name}' to '{dest_path_final}'", detail_level=2)
                        continue
                    else:
                        log_progress(f"Rsync failed for directory '{item_name}', falling back to shutil.copytree", detail_level=2)
                shutil.copytree(src_path, dest_path_final, dirs_exist_ok=True)
                log_progress(f"Copied directory (shutil): '{item_name}' to '{dest_path_final}'", detail_level=2)
            else:
                shutil.copy2(src_path, dest_path_final)
                log_progress(f"Copied file: '{item_name}' to '{dest_path_final}'", detail_level=2)
        else:
            log_progress(f"Warning: Source item not found for copy: '{src_path}'", detail_level=2)

    log_progress("Completed file transfer attempt.")


def run_configuration(config_dict: Dict[str, Union[str, int]],
                      all_datasets_dict: Dict[str, pd.DataFrame],
                      dataset_subsamples_info: Dict[str, Dict[str, str]],
                      global_temp_dir: str,
                      global_output_dir: str) -> str:
    """Run one configuration (dataset x k [x linkage])."""
    dataset_name = str(config_dict["dataset"])  # type: ignore
    num_clusters = int(config_dict["n_clusters"])  # type: ignore
    method_name = str(config_dict["method"])  # type: ignore
    linkage_name = str(config_dict.get("linkage", "")) if method_name == "agglomerative" else ""

    log_progress(f"\n=== Starting configuration: dataset={dataset_name}, k={num_clusters}, method={method_name}, linkage={linkage_name or '-'} ===")

    samples_df = all_datasets_dict[dataset_name]
    samples_array = prepare_samples_array(samples_df)

    num_samples_total = samples_array.shape[0]
    num_features_dim = samples_array.shape[1]
    train_subsample_size = int(num_samples_total * DEFAULT_CONFIG['PERCENT_SUBSAMPLE'])
    test_subsample_size = num_samples_total - train_subsample_size

    if method_name == "kmeans":
        method_str_val = "KMeans"
    elif method_name == "agglomerative":
        if linkage_name not in ["single", "ward"]:
            raise ValueError(f"Invalid linkage: {linkage_name}. Expected one of ['single', 'ward'].")
        method_str_val = f"AC_{linkage_name}"
    else:
        raise ValueError(f"Invalid method: {method_name}")

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name_suffix = f"{dataset_name}_n{num_samples_total}_clusters{num_clusters}_{method_str_val}_NoDimReduction_{timestamp_str}"

    temp_save_folder_this_run = os.path.join(global_temp_dir, run_name_suffix)
    final_save_folder_this_run = os.path.join(global_output_dir, run_name_suffix)
    os.makedirs(temp_save_folder_this_run, exist_ok=True)
    log_progress(f"Temporary directory for this run: {temp_save_folder_this_run}")

    method_cluster_temp_path = os.path.join(temp_save_folder_this_run, f'{method_str_val}_{num_clusters}_clusters')
    os.makedirs(method_cluster_temp_path, exist_ok=True)

    samples_original_csv_path = os.path.join(temp_save_folder_this_run, 'samples_original.csv')
    pd.DataFrame(samples_array).to_csv(samples_original_csv_path, index=False, header=False)
    log_progress(f"Original samples saved to {samples_original_csv_path}")

    subsamples_data_path = dataset_subsamples_info[dataset_name]['folder']
    indices_data_path = dataset_subsamples_info[dataset_name]['indices_folder']

    all_train_indices_list = np.load(os.path.join(indices_data_path, 'all_train_indices.npy'), allow_pickle=True)
    all_test_indices_list = np.load(os.path.join(indices_data_path, 'all_test_indices.npy'), allow_pickle=True)
    log_progress(f"Using precomputed train/test indices from .npy files in {indices_data_path}")

    # 1. Global centroids
    if method_name == "kmeans":
        global_sorted_centroids_arr = calculate_and_save_sorted_centroids_kmeans(
            samples_array, num_clusters, method_str_val, temp_save_folder_this_run
        )
    else:  # agglomerative
        global_sorted_centroids_arr = calculate_and_save_sorted_centroids_hc(
            samples_array, num_clusters, method_str_val, linkage_name, temp_save_folder_this_run
        )

    # 2. Per-iteration clustering + centroids (in-memory)
    if method_name == "kmeans":
        unaligned_predictions_matrix, iteration_centroids_list = perform_kmeans_clustering_and_get_centroids(
            DEFAULT_CONFIG['B'], num_clusters,
            subsamples_data_path,
            method_cluster_temp_path,
            temp_save_folder_this_run,
            method_str_val,
            test_subsample_size,
        )
    else:
        unaligned_predictions_matrix, iteration_centroids_list = perform_hc_clustering_and_get_centroids(
            DEFAULT_CONFIG['B'], num_clusters, linkage_name,
            subsamples_data_path,
            method_cluster_temp_path,
            temp_save_folder_this_run,
            method_str_val,
            test_subsample_size,
        )

    # 3. Align to global centroids
    if method_name == "kmeans":
        aligned_matrix = align_cluster_identities_kmeans(
            DEFAULT_CONFIG['B'], test_subsample_size, num_clusters,
            global_sorted_centroids_arr,
            unaligned_predictions_matrix,
            iteration_centroids_list,
            method_cluster_temp_path,
            method_str_val,
            temp_save_folder_this_run,
        )
    else:
        aligned_matrix = align_cluster_identities_hc(
            DEFAULT_CONFIG['B'], test_subsample_size, num_clusters,
            global_sorted_centroids_arr,
            unaligned_predictions_matrix,
            iteration_centroids_list,
            method_cluster_temp_path,
            method_str_val,
            temp_save_folder_this_run,
        )

    # Save aligned matrices
    aligned_matrix_path = os.path.join(temp_save_folder_this_run, 'aligned_matrix.npy')
    np.save(aligned_matrix_path, aligned_matrix)
    log_progress(f"Aligned matrix saved to {aligned_matrix_path}")

    aligned_predictions_csv_path = os.path.join(temp_save_folder_this_run, f'{method_str_val}_predicted_testdata_80_20_aligned.csv')
    pd.DataFrame(aligned_matrix).to_csv(aligned_predictions_csv_path, index=False, header=False)
    log_progress(f"Aligned predictions CSV saved to {aligned_predictions_csv_path}")

    # 4. Final CLAM matrix
    if method_name == "kmeans":
        _clam_matrix_final = generate_clam_matrix_kmeans(
            samples_array, all_test_indices_list,
            aligned_matrix, num_clusters,
            method_cluster_temp_path,
            dataset_name=dataset_name,
            method_str_val_param=method_str_val,
        )
    else:
        _clam_matrix_final = generate_clam_matrix_hc(
            samples_array, all_test_indices_list,
            aligned_matrix, num_clusters,
            method_cluster_temp_path,
            dataset_name=dataset_name,
            method_str_val_param=method_str_val,
        )

    # Copy CLAM to output root for convenience
    clam_filename = f'clam_{dataset_name}_k{num_clusters}_{method_str_val}.csv'
    temp_clam_path = os.path.join(method_cluster_temp_path, clam_filename)
    final_clam_path = None
    if os.path.exists(temp_clam_path):
        final_clam_path = os.path.join(global_output_dir, clam_filename)
        shutil.copy2(temp_clam_path, final_clam_path)
        log_progress(f"Copied CLAM file for easy access to: {final_clam_path}", detail_level=2)

    # Save run metadata
    config_info_dict = {
        'timestamp': timestamp_str,
        'parameters': {
            'dataset': dataset_name,
            'n_clusters': num_clusters,
            'method': method_name,
            'linkage': linkage_name if method_name == 'agglomerative' else None,
            'B': DEFAULT_CONFIG['B'],
            'n_samples': num_samples_total,
            'dimensionality': num_features_dim,
            'train_percent': DEFAULT_CONFIG['PERCENT_SUBSAMPLE'],
            'seed': DEFAULT_CONFIG['RANDOM_SEED'],
            'data_path': DEFAULT_CONFIG['DATA_PATH'],
            'output_dir_base': global_output_dir,
            'temp_dir_base': global_temp_dir,
            'final_run_output_folder': final_save_folder_this_run,
        },
        'file_paths_within_temp_run_folder': {
            'samples_original': samples_original_csv_path,
            'method_cluster_folder': method_cluster_temp_path,
            'global_sorted_centroids_csv': os.path.join(method_cluster_temp_path, 'centroid_values_sorted.csv'),
            'aligned_matrix_npy': os.path.join(temp_save_folder_this_run, 'aligned_matrix.npy'),
            'aligned_predictions_csv': os.path.join(temp_save_folder_this_run, f'{method_str_val}_predicted_testdata_80_20_aligned.csv'),
            'final_clam_csv': temp_clam_path,
            'final_clam_npy': os.path.join(method_cluster_temp_path, 'clam_matrix.npy') if method_name == 'kmeans' else None,
            'easy_access_clam_path': final_clam_path,
        }
    }
    yaml_path = os.path.join(temp_save_folder_this_run, 'config_info.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config_info_dict, f, sort_keys=False)

    log_progress(f"Configuration processing complete. Temp results in {temp_save_folder_this_run}")

    log_progress("Copying results to final destination")
    copy_results_to_destination(temp_save_folder_this_run, final_save_folder_this_run, method_str_val, config_dict)

    log_progress(f"=== Completed configuration. Final results in: {final_save_folder_this_run} ===")
    return final_save_folder_this_run


def run_mcss_analysis(config_payload: Dict[str, Union[str, int]],
                     ui_placeholders: Dict[str, st.empty]) -> List[str]:
    """Run MCSS analysis on uploaded files."""
    # Reset logs and status
    st.session_state.log_messages = []
    st.session_state.run_status['last_error'] = ''

    final_folders: List[str] = []
    
    try:
        if not st.session_state.uploaded_files_info:
            error_msg = "Please upload at least one dataset file (.npy or .csv)"
            log_progress(error_msg)
            st.toast("🚨 Please upload at least one dataset file!", icon="📤")
            return []

        # Process uploaded files
        selected_datasets = []
        for uploaded_file in st.session_state.uploaded_files_info:
            try:
                dataset_name, filename = process_streamlit_file(uploaded_file)
                selected_datasets.append({'name': dataset_name, 'filename': filename})
                log_progress(f"Processed uploaded file: {dataset_name}")
            except Exception as e:
                log_progress(f"Error processing uploaded file: {e}")
                continue

        if not selected_datasets:
            error_msg = "No valid dataset files could be processed"
            log_progress(error_msg)
            st.toast("🚨 No valid dataset files could be processed.", icon="❌")
            return []

        # Update config
        DEFAULT_CONFIG['SELECTED_DATASETS'] = selected_datasets
        DEFAULT_CONFIG['METHOD'] = config_payload.get("methods", ["kmeans", "agglomerative"])
        DEFAULT_CONFIG['LINKAGES'] = config_payload.get("linkages", ["single", "ward"])
        DEFAULT_CONFIG['CLUSTER_RANGE'] = list(range(int(config_payload.get("k_min", 2)), int(config_payload.get("k_max", 5)) + 1))
        DEFAULT_CONFIG['B'] = int(config_payload.get("B", 200))
        DEFAULT_CONFIG['PERCENT_SUBSAMPLE'] = float(config_payload.get("percent_subsample", 0.8))
        DEFAULT_CONFIG['RANDOM_SEED'] = int(config_payload.get("seed", 123))
        if config_payload.get("output_dir") and isinstance(config_payload["output_dir"], str) and config_payload["output_dir"].strip():
            DEFAULT_CONFIG['OUTPUT_DIR'] = config_payload["output_dir"].strip()

        # Set deterministic mode
        set_deterministic_mode(int(DEFAULT_CONFIG['RANDOM_SEED']))

        # Setup directories
        temp_dir_base = DEFAULT_CONFIG['TEMP_DIR']
        output_dir_base = DEFAULT_CONFIG['OUTPUT_DIR']
        data_path = DEFAULT_CONFIG['DATA_PATH']

        os.makedirs(temp_dir_base, exist_ok=True)
        os.makedirs(output_dir_base, exist_ok=True)

        run_date = datetime.now().strftime('%Y%m%d')
        dated_output_dir = os.path.join(output_dir_base, run_date)
        os.makedirs(dated_output_dir, exist_ok=True)

        # Load datasets
        log_progress("Loading uploaded datasets")
        loaded_datasets = load_raw_data(selected_datasets, data_path)
        if not loaded_datasets:
            log_progress("Could not load any datasets.")
            st.toast("🚨 Error: Could not load datasets", icon="❌")
            return []

        # Precompute subsamples
        log_progress("Precomputing Monte Carlo subsamples (shared per dataset)")
        dataset_subsamples_storage: Dict[str, Dict[str, str]] = {}
        percent_train = DEFAULT_CONFIG['PERCENT_SUBSAMPLE']
        
        for dataset_id in loaded_datasets:
            log_progress(f"Preparing subsamples for dataset: {dataset_id}", detail_level=2)
            current_samples_df = loaded_datasets[dataset_id]
            _samples_arr = prepare_samples_array(current_samples_df)
            _n_total = _samples_arr.shape[0]
            _n_train = int(_n_total * percent_train)
            
            shared_subsample_loc = os.path.join(temp_dir_base, f"{dataset_id}_shared_subsamples")
            expected_indices_folder = os.path.join(shared_subsample_loc, 'indices')
            expected_train_indices_file = os.path.join(expected_indices_folder, 'all_train_indices.npy')
            expected_test_indices_file = os.path.join(expected_indices_folder, 'all_test_indices.npy')
            expected_subsamples_data_folder = os.path.join(shared_subsample_loc, 'subsamples_data')
            
            if (not os.path.exists(expected_train_indices_file) or
                not os.path.exists(expected_test_indices_file) or
                not os.path.exists(expected_subsamples_data_folder) or
                not os.path.exists(os.path.join(expected_subsamples_data_folder, 'subset_data_0.npy'))):
                
                log_progress(f"Required subsample files not found. Regenerating.", detail_level=2)
                if os.path.exists(shared_subsample_loc):
                    try:
                        shutil.rmtree(shared_subsample_loc)
                    except OSError as e:
                        log_progress(f"Warning: Could not remove directory {shared_subsample_loc}: {e}. Trying alternative cleanup.", detail_level=2)
                        # Try to remove contents recursively with force
                        import stat
                        def handle_remove_readonly(func, path, exc):
                            if os.path.exists(path):
                                os.chmod(path, stat.S_IWRITE)
                                func(path)
                        try:
                            shutil.rmtree(shared_subsample_loc, onerror=handle_remove_readonly)
                        except OSError:
                            log_progress(f"Warning: Could not completely remove {shared_subsample_loc}. Continuing with existing directory.", detail_level=2)
                os.makedirs(shared_subsample_loc, exist_ok=True)
                subsamples_loc_path, indices_loc_path = monte_carlo_subsampling(_samples_arr, _n_total, DEFAULT_CONFIG['B'], _n_train, shared_subsample_loc)
            else:
                log_progress(f"Reusing existing subsamples from {shared_subsample_loc}", detail_level=2)
                subsamples_loc_path = expected_subsamples_data_folder
                indices_loc_path = expected_indices_folder
            
            dataset_subsamples_storage[dataset_id] = {
                'folder': subsamples_loc_path,
                'indices_folder': indices_loc_path,
            }

        # Build configurations
        selected_method = DEFAULT_CONFIG['METHOD']
        linkages_local = DEFAULT_CONFIG.get('LINKAGES') or ["single", "ward"]
        cluster_values = DEFAULT_CONFIG['CLUSTER_RANGE']
        
        configurations: List[Dict[str, Union[str, int]]] = []
        if selected_method == 'kmeans':
            for dataset_info in DEFAULT_CONFIG['SELECTED_DATASETS']:
                for k in cluster_values:
                    configurations.append({'dataset': dataset_info['name'], 'n_clusters': int(k), 'method': 'kmeans'})
        elif selected_method == 'agglomerative':
            for dataset_info in DEFAULT_CONFIG['SELECTED_DATASETS']:
                for k in cluster_values:
                    for linkage_name in linkages_local:
                        configurations.append({'dataset': dataset_info['name'], 'n_clusters': int(k), 'method': 'agglomerative', 'linkage': linkage_name})
        elif selected_method == 'both':
            for dataset_info in DEFAULT_CONFIG['SELECTED_DATASETS']:
                for k in cluster_values:
                    configurations.append({'dataset': dataset_info['name'], 'n_clusters': int(k), 'method': 'kmeans'})
            for dataset_info in DEFAULT_CONFIG['SELECTED_DATASETS']:
                for k in cluster_values:
                    for linkage_name in linkages_local:
                        configurations.append({'dataset': dataset_info['name'], 'n_clusters': int(k), 'method': 'agglomerative', 'linkage': linkage_name})
        
        reset_run_status(len(configurations))
        st.session_state.run_status['phase'] = 'Starting execution'

        # Execute configurations
        random.seed(DEFAULT_CONFIG['RANDOM_SEED'])
        np.random.seed(DEFAULT_CONFIG['RANDOM_SEED'])
        
        progress_placeholder = ui_placeholders["progress_bar"]
        status_text_placeholder = ui_placeholders["status_text"]
        log_box_placeholder = ui_placeholders["log_box"]
        results_box_placeholder = ui_placeholders["results_box"]

        for i, current_config in enumerate(configurations):
            if st.session_state.cancel_requested:
                log_progress("Run cancelled by user")
                break
                
            st.session_state.run_status['current_index'] = i + 1
            dataset_name = current_config['dataset']
            method_name = current_config['method']
            k_val = current_config['n_clusters']
            linkage_str = f" ({current_config.get('linkage', '')})" if current_config.get('linkage') else ""
            st.session_state.run_status['phase'] = f"Processing {dataset_name}: {method_name}{linkage_str}, k={k_val}"
            
            # Update progress in UI
            progress = overall_progress()
            eta = eta_text()
            progress_placeholder.progress(progress, text=f"Processing configuration {i+1}/{len(configurations)}: {dataset_name} - {method_name}{linkage_str}, k={k_val} | {eta}")
            status_text_placeholder.text(f"Processing configuration {i+1}/{len(configurations)}: {dataset_name} - {method_name}{linkage_str}, k={k_val} | {eta}")
            
            log_progress(f"\n=== Configuration {i+1}/{len(configurations)} ===")
            log_progress(f"Dataset: {dataset_name}, Method: {method_name}{linkage_str}, k={k_val}")
            try:
                saved_to_folder = run_configuration(current_config, loaded_datasets, dataset_subsamples_storage, temp_dir_base, dated_output_dir)
                final_folders.append(saved_to_folder)
                st.session_state.run_status['output_folders'] = final_folders  # Update session state
                log_progress(f"Successfully completed configuration {i+1}. Results in {saved_to_folder}", detail_level=2)
            except Exception as e_run:
                log_progress(f"ERROR processing configuration {i+1} ({current_config}): {str(e_run)}")
                import traceback
                log_progress(traceback.format_exc())
                continue

        # Final status update
        st.session_state.run_status['phase'] = 'Completed'
        st.session_state.run_status['current_index'] = len(configurations)
        st.session_state.run_status['is_running'] = False
        progress_placeholder.progress(1.0, text="✅ Analysis completed!")
        status_text_placeholder.text("✅ Analysis completed!")

        log_progress(f"Analysis completed. {len(final_folders)} configurations processed successfully.")
        return final_folders

    except Exception as e:
        error_msg = f"FATAL ERROR during run: {e}"
        log_progress(error_msg)
        st.session_state.run_status['last_error'] = f"ERROR: {e}"
        st.session_state.run_status['is_running'] = False
        st.toast(f"❌ Analysis failed: {e}", icon="❌")
        import traceback
        log_progress(traceback.format_exc())
        return []


######################
# Streamlit UI
######################

def main():
    """Main Streamlit application."""
    if not _STREAMLIT_AVAILABLE:
        st.error("Streamlit is not installed. Please install it with 'pip install streamlit'.")
        return

    init_session_state()

    st.set_page_config(
        page_title="MCSS Clustering Analysis",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧬 MCSS Clustering Analysis")
    st.markdown("*Monte Carlo Subsampling for Clustering Replicability*")
    
    # --- UI LAYOUT ---
    st.markdown("---")
    
    # Status & Progress Bar
    st.header("Status & Progress")
    progress_bar = st.progress(0, text="Ready")
    status_text = st.empty()
    st.markdown("---")

    col_config, col_main = st.columns(2)

    with col_config:
        st.header("1. Configure Parameters")
        
        with st.container(border=True):
            st.subheader("Select Methods")
            methods = st.multiselect("Clustering Methods", ["kmeans", "agglomerative"], default=["kmeans", "agglomerative"])
            linkages = []
            if 'agglomerative' in methods:
                linkages = st.multiselect("Linkages (for Agglomerative)", ["single", "ward"], default=["single", "ward"])

        with st.container(border=True):
            st.subheader("Configure Parameters")
            k_min = st.number_input("k min", min_value=2, value=2, key="k_min")
            k_max = st.number_input("k max", min_value=2, value=5, key="k_max")
            B_in = st.number_input("Iterations (B)", min_value=10, value=200, key="B_in")
            percent_in = st.slider("Train %", 0.1, 0.95, 0.8, 0.05, key="percent_in")

        with st.expander("Advanced"):
            seed_in = st.number_input("Random Seed", value=123, key="seed_in")

    with col_main:
        st.header("2. Upload & Run")
        
        with st.container(border=True):
            st.subheader("Upload Datasets")
            uploaded_files = st.file_uploader(
                "Upload Datasets (.npy/.csv)", 
                accept_multiple_files=True, 
                type=['npy', 'csv'],
                key="file_uploader"
            )
            if uploaded_files:
                st.session_state.uploaded_files_info = uploaded_files
                st.success(f"{len(uploaded_files)} file(s) ready for analysis.")

        with st.container(border=True):
            st.subheader("Actions")
            
            run_button = st.button("▶ Run Analysis", type="primary", use_container_width=True, disabled=not uploaded_files)
            
            # Download button - always visible but disabled when no results
            has_results = bool(st.session_state.run_status['output_folders'])
            
            if has_results:
                zip_data = create_results_zip(st.session_state.run_status['output_folders'])
                if zip_data:
                    st.download_button(
                        label="📥 Download Results ZIP",
                        data=zip_data,
                        file_name=f"mcss_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        help=f"Download {len(st.session_state.run_status['output_folders'])} result folder(s) as a ZIP file"
                    )
                else:
                    st.button("📥 Download Results ZIP", disabled=True, use_container_width=True, help="Error creating download package")
            else:
                st.button("📥 Download Results ZIP", disabled=True, use_container_width=True, help="Run analysis first to generate results")

    # --- RUN LOGIC ---
    if run_button:
        if not st.session_state.uploaded_files_info:
            st.toast("🚨 Please upload at least one dataset file!", icon="📤")
        else:
            # Assemble config from UI controls
            config_payload = {
                "methods": methods,
                "linkages": linkages,
                "k_min": k_min,
                "k_max": k_max,
                "B": B_in,
                "percent_subsample": percent_in,
                "seed": seed_in
            }
            # Define UI placeholders to update during the run
            ui_placeholders = {
                "progress_bar": progress_bar,
                "status_text": status_text,
                "log_box": st.empty(), # Changed from log_col to st.empty()
                "results_box": st.empty() # Changed from res_col to st.empty()
            }
            run_mcss_analysis(config_payload, ui_placeholders)

    # --- RESULTS & LOGS (Static Display) ---
    st.markdown("---")
    
    res_col, log_col = st.columns(2)
    with res_col:
        st.header("Results")
        with st.container(border=True, height=300):
            if st.session_state.run_status['output_folders']:
                st.write("Analysis Complete. Output folders:")
                for folder in st.session_state.run_status['output_folders']:
                    st.code(folder, language=None)
                
                # Add download button in results section
                st.markdown("---")
                zip_data = create_results_zip(st.session_state.run_status['output_folders'])
                if zip_data:
                    st.download_button(
                        label="📥 Download All Results as ZIP",
                        data=zip_data,
                        file_name=f"mcss_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        type="primary"
                    )
                    st.success(f"✅ {len(st.session_state.run_status['output_folders'])} result folder(s) ready for download!")
                else:
                    st.error("❌ Error creating download package")
            else:
                st.info("Results will appear here after a successful run.")
    
    with log_col:
        st.header("Logs")
        with st.container(border=True, height=300):
            log_content = get_logs_for_ui()
            st.code(log_content, language='log')

if __name__ == "__main__":
    main()

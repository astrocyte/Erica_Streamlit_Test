# MCSS Streamlit Application

A Streamlit version of the Monte Carlo Subsampling for Clustering Replicability (MCSS) script that supports direct file upload and real-time progress tracking.

## Features

- **Direct File Upload**: Upload .npy or .csv dataset files directly through the web interface
- **Multiple Methods**: Support for K-means and Agglomerative clustering
- **Deterministic Results**: Reproducible results with seed control
- **Real-time Progress**: Live updates of progress and execution logs
- **CLAM Matrix Generation**: Creates Cluster Co-occurrence Likeliness matrices
- **Download Results**: One-click download of all results as a zip file
- **Interactive UI**: Clean, responsive Streamlit interface

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run mcss_streamlit_app.py
```

## Usage

1. **Configure Parameters**: Use the sidebar to set clustering parameters:
   - Clustering method (K-means, Agglomerative, or Both)
   - Range of k values to evaluate
   - Linkage criteria for agglomerative clustering
   - Training percentage, iterations, random seed
   - Output directory

2. **Upload Files**: In the "Upload & Run" tab:
   - Upload one or more dataset files (.npy or .csv format)
   - Files are processed automatically upon upload

3. **Run Analysis**: Click "Run MCSS Analysis" to start the clustering analysis

4. **Monitor Progress**: Switch to the "Progress" tab to:
   - View real-time progress updates
   - Monitor execution logs
   - See current processing status

5. **Download Results**: Once complete, download all results as a zip file

## Supported File Formats

- **.npy files**: NumPy arrays (can contain dictionaries with 'all' key)
- **.csv files**: Comma-separated values (Gene-by-Sample format expected)

## Output

The application generates:
- Clustering results and CLAM matrices
- Detailed logs and configuration files
- Individual result folders for each configuration
- Ready for ERICA metrics analysis

## Configuration

Default settings can be modified in the `DEFAULT_CONFIG` dictionary at the top of the script:

```python
DEFAULT_CONFIG = {
    'B': 200,  # Number of Monte Carlo iterations
    'PERCENT_SUBSAMPLE': 0.8,  # Proportion of data for training subsample
    'OUTPUT_DIR': "MCSS_Streamlit_Output",  # Base directory for output
    'RANDOM_SEED': 123,
    'CLUSTER_RANGE': list(range(2, 6)),  # Range of cluster numbers to evaluate
    'METHOD': 'both',  # 'kmeans', 'agglomerative', or 'both'
    'LINKAGES': ["single", "ward"],  # For agglomerative clustering
}
```

## Comparison with Gradio Version

This Streamlit version provides:
- More structured UI with tabs and sidebar configuration
- Better progress tracking with session state management
- More intuitive file upload and download experience
- Cleaner separation of configuration and execution phases
- Better error handling and user feedback

## Troubleshooting

If you encounter issues:

1. **Import Errors**: Ensure all dependencies are installed
2. **File Upload Issues**: Check file format (.npy or .csv) and size
3. **Memory Issues**: Reduce the number of iterations or dataset size
4. **Permission Issues**: Ensure write access to the output directory

## Development

The application is structured as follows:
- Configuration and utility functions
- Core clustering algorithms (K-means and Agglomerative)
- File processing and data loading functions
- Main orchestration and execution logic
- Streamlit UI components and interaction handlers

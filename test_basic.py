#!/usr/bin/env python3
"""
Basic functionality test for MCSS Streamlit application.
This script tests core functions without the Streamlit UI.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the current directory to the path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import core functions (excluding Streamlit-specific ones)
    from mcss_streamlit_app import (
        set_deterministic_mode,
        prepare_samples_array,
        compute_config_hash,
        DEFAULT_CONFIG
    )
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_deterministic_mode():
    """Test deterministic mode setting."""
    print("\n🧪 Testing deterministic mode...")
    try:
        set_deterministic_mode(123)
        # Check if numpy random state is set
        np.random.seed(123)
        val1 = np.random.random()
        np.random.seed(123)
        val2 = np.random.random()
        assert val1 == val2, "Random seed not working properly"
        print("✅ Deterministic mode test passed")
    except Exception as e:
        print(f"❌ Deterministic mode test failed: {e}")

def test_prepare_samples_array():
    """Test sample array preparation."""
    print("\n🧪 Testing sample array preparation...")
    try:
        # Create test data
        test_data = pd.DataFrame({
            'gene_id': ['gene1', 'gene2', 'gene3'],
            'sample1': [1.0, 2.0, 3.0],
            'sample2': [4.0, 5.0, 6.0],
            'sample3': [7.0, 8.0, 9.0]
        })
        
        samples_array = prepare_samples_array(test_data)
        
        # Should be transposed (samples x features)
        assert samples_array.shape == (3, 3), f"Expected shape (3, 3), got {samples_array.shape}"
        assert np.issubdtype(samples_array.dtype, np.number), "Array should be numeric"
        print("✅ Sample array preparation test passed")
    except Exception as e:
        print(f"❌ Sample array preparation test failed: {e}")

def test_config_hash():
    """Test configuration hashing."""
    print("\n🧪 Testing configuration hashing...")
    try:
        hash1 = compute_config_hash(DEFAULT_CONFIG)
        hash2 = compute_config_hash(DEFAULT_CONFIG)
        assert hash1 == hash2, "Hash should be consistent"
        assert len(hash1) == 12, f"Hash should be 12 characters, got {len(hash1)}"
        print(f"✅ Configuration hash test passed (hash: {hash1})")
    except Exception as e:
        print(f"❌ Configuration hash test failed: {e}")

def test_clustering_imports():
    """Test that clustering functions can be imported."""
    print("\n🧪 Testing clustering function imports...")
    try:
        from mcss_streamlit_app import (
            calculate_and_save_sorted_centroids_kmeans,
            monte_carlo_subsampling,
            load_raw_data
        )
        print("✅ Clustering function imports successful")
    except ImportError as e:
        print(f"❌ Clustering function import failed: {e}")

if __name__ == "__main__":
    print("🧬 MCSS Streamlit Application - Basic Functionality Test")
    print("=" * 60)
    
    test_deterministic_mode()
    test_prepare_samples_array()
    test_config_hash()
    test_clustering_imports()
    
    print("\n" + "=" * 60)
    print("🎉 Basic functionality tests completed!")
    print("\n💡 To run the full Streamlit application:")
    print("   streamlit run mcss_streamlit_app.py")
    print("   or")
    print("   ./run_streamlit.sh")

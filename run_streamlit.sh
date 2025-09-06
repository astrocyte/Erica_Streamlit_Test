#!/bin/bash

# MCSS Streamlit Application Launcher
# This script launches the Streamlit version of the MCSS clustering tool

echo "🧬 Starting MCSS Streamlit Application..."
echo "📋 Make sure you have installed the requirements:"
echo "   pip install -r requirements.txt"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed. Please install it with:"
    echo "   pip install streamlit"
    exit 1
fi

# Launch the Streamlit app
echo "🚀 Launching Streamlit app..."
streamlit run mcss_streamlit_app.py --server.maxUploadSize=1000

echo ""
echo "📝 The application should now be available in your web browser."
echo "   If it doesn't open automatically, go to: http://localhost:8501"
echo ""
echo "💡 Tips:"
echo "   - Upload .npy or .csv files in the 'Upload & Run' tab"
echo "   - Monitor progress in the 'Progress' tab"
echo "   - Download results when analysis is complete"

"""
PART 2: MAIN APPLICATION - app.py
==================================

This is the MAIN file that runs your entire application!
Save this as 'app.py' in your landcover_app folder.

To run the app, open terminal and type:
    streamlit run app.py

Then a browser window will open automatically!
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add our utils folder to Python's search path
sys.path.append(str(Path(__file__).parent))

# Import our helper functions
from utils.helpers import create_folder, save_config, load_config

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Land Cover Classification System",
    page_icon="🛰️",
    layout="wide",  # Use full screen width
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
"""
Session state is like the app's memory - it remembers things even when 
you click buttons and the page refreshes!
"""

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Current step in the workflow
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'welcome'
    
    # User's choice: existing or custom dataset
    if 'dataset_choice' not in st.session_state:
        st.session_state.dataset_choice = None
    
    # Configuration for custom dataset
    if 'region_geometry' not in st.session_state:
        st.session_state.region_geometry = None
    
    if 'timeline_config' not in st.session_state:
        st.session_state.timeline_config = {}
    
    if 'preprocessing_config' not in st.session_state:
        st.session_state.preprocessing_config = {}
    
    # Downloaded/loaded data
    if 'raw_data_paths' not in st.session_state:
        st.session_state.raw_data_paths = []
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Data splits
    if 'train_data' not in st.session_state:
        st.session_state.train_data = None
    if 'val_data' not in st.session_state:
        st.session_state.val_data = None
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    
    # Split configuration
    if 'split_config' not in st.session_state:
        st.session_state.split_config = {}

initialize_session_state()

# ============================================================================
# SIDEBAR - NAVIGATION & PROGRESS
# ============================================================================

def show_sidebar():
    """Show sidebar with progress and navigation"""
    
    with st.sidebar:
        st.title("🛰️ Land Cover App")
        st.markdown("---")
        
        # Progress tracker
        st.subheader("📍 Progress")
        
        steps = {
            'welcome': '1️⃣ Welcome',
            'dataset_choice': '2️⃣ Choose Dataset',
            'load_existing': '3️⃣ Load Data',
            'region_selection': '3️⃣ Select Region',
            'timeline_config': '4️⃣ Configure Timeline',
            'preprocessing': '5️⃣ Preprocessing',
            'downloading': '6️⃣ Download Data',
            'data_splitting': '7️⃣ Split Data',
            'ml_preparation': '8️⃣ Prepare for ML',
            'model_training': '9️⃣ Train Model'
        }
        
        current = st.session_state.current_step
        
        for key, label in steps.items():
            if key == current:
                st.markdown(f"**➡️ {label}** ⬅️")
            else:
                st.markdown(f"   {label}")
        
        st.markdown("---")
        
        # Quick stats if data is loaded
        if st.session_state.processed_data is not None:
            st.subheader("📊 Dataset Info")
            data = st.session_state.processed_data
            st.metric("Total Samples", len(data))
            if 'label' in data.columns:
                st.metric("Number of Classes", data['label'].nunique())
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

show_sidebar()

# ============================================================================
# STEP 1: WELCOME SCREEN
# ============================================================================

def show_welcome_screen():
    """
    The first screen users see - explains what the app does
    """
    
    st.title("🛰️ Welcome to Land Cover Classification System")
    
    st.markdown("""
    ### What does this app do?
    
    This application helps you create datasets for **land cover classification** 
    using satellite imagery. You can:
    
    - 🌍 Download satellite images from Google Earth Engine
    - 🏷️ Use pre-labeled datasets or create your own
    - 📊 Split data into training/validation/test sets
    - 🤖 Prepare data for machine learning models
    
    ### What is Land Cover Classification?
    
    It's like teaching a computer to look at satellite images and identify:
    - 🌳 **Green areas** (forests, farms, parks)
    - 💧 **Water bodies** (rivers, lakes, oceans)
    - 🏗️ **Built-up areas** (cities, buildings, roads)
    - 🏜️ **Bare land** (deserts, mountains, empty land)
    """)
    
    st.image("https://www.researchgate.net/publication/340784693/figure/fig1/AS:882291997315072@1587386646382/Land-cover-classification-workflow.png", 
             caption="Example: Satellite image → Classification", 
             use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🚀 Ready to start?")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("▶️ Let's Begin!", type="primary", use_container_width=True):
            st.session_state.current_step = 'dataset_choice'
            st.rerun()

# ============================================================================
# STEP 2: DATASET CHOICE
# ============================================================================

def show_dataset_choice():
    """
    Let user choose between existing dataset or creating custom one
    """
    
    st.title("📂 Choose Your Dataset Option")
    
    st.markdown("""
    You have two options to get started. Choose the one that fits your needs:
    """)
    
    col1, col2 = st.columns(2)
    
    # Option 1: Use Existing Dataset
    with col1:
        st.markdown("### 📚 Option 1: Use Existing Dataset")
        
        with st.container(border=True):
            st.markdown("""
            **IndiaSAT Dataset**
            
            ✅ **Quick Start** - Ready to use immediately
            
            📊 **Pre-labeled** - 180,000 labeled pixels
            
            🗺️ **Multiple Regions** - Various locations across India
            
            🏷️ **4 Classes**:
            - Green (vegetation)
            - Water (rivers, lakes)
            - Bare (soil, rocks)
            - Built (buildings, roads)
            
            📅 **Timeline**: 2013-2019
            
            🔬 **Quality**: Manually verified labels
            """)
            
            st.markdown("**Best for:**")
            st.info("🎓 Learning, quick experiments, benchmarking")
            
            if st.button("Use Existing Dataset", type="primary", use_container_width=True):
                st.session_state.dataset_choice = 'existing'
                st.session_state.current_step = 'load_existing'
                st.rerun()
    
    # Option 2: Create Custom Dataset
    with col2:
        st.markdown("### 🎨 Option 2: Create Custom Dataset")
        
        with st.container(border=True):
            st.markdown("""
            **Build Your Own**
            
            🗺️ **Custom Region** - Choose any location
            
            📅 **Flexible Timeline** - Any date range
            
            🛰️ **Satellite Choice** - Landsat or Sentinel
            
            ⚙️ **Full Control**:
            - Select specific bands
            - Calculate indices (NDVI, NDWI)
            - Custom preprocessing
            - Cloud masking options
            
            🎯 **Your Classes** - Define what you want to classify
            """)
            
            st.markdown("**Best for:**")
            st.info("🔬 Research, specific regions, custom requirements")
            
            if st.button("Create Custom Dataset", use_container_width=True):
                st.session_state.dataset_choice = 'custom'
                st.session_state.current_step = 'region_selection'
                st.rerun()
    
    # Comparison table
    st.markdown("---")
    st.subheader("📊 Quick Comparison")
    
    comparison_data = {
        "Feature": ["Setup Time", "Flexibility", "Data Quality", "Region Coverage", "Best For"],
        "Existing Dataset": ["⚡ Instant", "🔒 Fixed", "✅ Verified", "🇮🇳 India", "🎓 Learning"],
        "Custom Dataset": ["⏱️ 15-30 min", "🎨 Full Control", "❓ Depends", "🌍 Worldwide", "🔬 Research"]
    }
    
    import pandas as pd
    st.table(pd.DataFrame(comparison_data))
    
    st.markdown("---")
    
    # Back button
    if st.button("← Back to Welcome"):
        st.session_state.current_step = 'welcome'
        st.rerun()

# ============================================================================
# MAIN APP LOGIC - ROUTING
# ============================================================================

def main():
    """
    Main function that controls which screen to show.
    Think of this as a traffic controller!
    """
    
    # Get current step from session state
    step = st.session_state.current_step
    
    # Show the appropriate screen based on current step
    if step == 'welcome':
        show_welcome_screen()
    
    elif step == 'dataset_choice':
        show_dataset_choice()
    
    elif step == 'load_existing':
        # Import and show the existing dataset loader
        from pages import load_existing_dataset
        load_existing_dataset.show()
    
    elif step == 'region_selection':
        # Import and show region selection
        from pages import region_selection
        region_selection.show()
    
    elif step == 'timeline_config':
        # Import and show timeline configuration
        from pages import timeline_config
        timeline_config.show()
    
    elif step == 'preprocessing':
        # Import and show preprocessing options
        from pages import preprocessing
        preprocessing.show()
    
    elif step == 'downloading':
        # Import and show download progress
        from pages import data_download
        data_download.show()
    
    elif step == 'data_splitting':
        # Import and show data splitting
        from pages import data_splitting
        data_splitting.show()
    
    elif step == 'ml_preparation':
        # Import and show ML preparation
        from pages import ml_preparation
        ml_preparation.show()
    
    elif step == 'model_training':
        # Import and show model training
        from pages import model_training
        model_training.show()

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()

"""
🎉 CONGRATULATIONS!

You've created the main application file!

TO RUN THE APP:
1. Open terminal in your landcover_app folder
2. Type: streamlit run app.py
3. Press Enter
4. Your browser will open automatically!

NEXT STEPS:
Now we need to create the individual page modules that this app imports.
These will go in a 'pages' folder:
- pages/load_existing_dataset.py
- pages/region_selection.py
- pages/timeline_config.py
- pages/preprocessing.py
- pages/data_download.py
- pages/data_splitting.py
- pages/ml_preparation.py
- pages/model_training.py

Go to Part 3 to create these page modules!
"""
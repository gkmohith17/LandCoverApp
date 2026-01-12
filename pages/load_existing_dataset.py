"""
PART 3A: LOAD EXISTING DATASET PAGE
====================================

Save this as: pages/load_existing_dataset.py

This page handles loading the IndiaSAT dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import json

def show():
    """Main function that shows this page"""
    
    st.title("📚 Load Existing IndiaSAT Dataset")
    
    st.markdown("""
    We'll now load the IndiaSAT dataset - a pre-labeled dataset of 180,000 pixels 
    from various regions across India.
    """)
    
    # ========================================================================
    # STEP 1: Check if IndiaSAT folder exists
    # ========================================================================
    
    st.subheader("Step 1: Check IndiaSAT Dataset")
    
    indiasat_path = Path("IndiaSAT")
    
    if not indiasat_path.exists():
        st.warning("⚠️ IndiaSAT dataset not found in your folder!")
        
        st.markdown("""
        ### 📥 How to download IndiaSAT:
        
        **Option 1: Using Git** (if you have Git installed)
        ```bash
        git clone https://github.com/ChahatBansal8060/IndiaSAT.git
        ```
        
        **Option 2: Manual Download** (easier for beginners)
        1. Go to: https://github.com/ChahatBansal8060/IndiaSAT
        2. Click the green "Code" button
        3. Click "Download ZIP"
        4. Extract the ZIP to your landcover_app folder
        5. Make sure the folder is named "IndiaSAT"
        
        After downloading, refresh this page!
        """)
        
        if st.button("🔄 Refresh Page"):
            st.rerun()
        
        return  # Stop here if dataset not found
    
    st.success("✅ IndiaSAT dataset found!")
    
    # ========================================================================
    # STEP 2: Scan available datasets
    # ========================================================================
    
    st.subheader("Step 2: Available Datasets")
    
    with st.spinner("Scanning for available datasets..."):
        datasets = scan_indiasat_datasets()
    
    if not datasets:
        st.error("❌ No datasets found in IndiaSAT folder!")
        st.info("Please check if the IndiaSAT folder structure is correct.")
        return
    
    st.success(f"✅ Found {len(datasets)} dataset(s)!")
    
    # Show dataset options
    st.markdown("### Choose a dataset:")
    
    for i, dataset_info in enumerate(datasets):
        with st.expander(f"📊 {dataset_info['name']}", expanded=(i==0)):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **📍 Region:** {dataset_info['region']}
                
                **📅 Years:** {dataset_info['years']}
                
                **🛰️ Satellite:** {dataset_info['satellite']}
                """)
            
            with col2:
                st.markdown(f"""
                **📊 Total Samples:** {dataset_info['samples']:,}
                
                **🏷️ Classes:** {dataset_info['classes']}
                
                **💾 Size:** {dataset_info['size_mb']} MB
                """)
            
            # Class distribution
            if dataset_info.get('class_distribution'):
                st.markdown("**Class Distribution:**")
                dist_df = pd.DataFrame({
                    'Class': list(dataset_info['class_distribution'].keys()),
                    'Count': list(dataset_info['class_distribution'].values())
                })
                st.bar_chart(dist_df.set_index('Class'))
            
            # Load button
            if st.button(f"Load This Dataset", key=f"load_{i}"):
                load_dataset(dataset_info)
    
    # ========================================================================
    # Navigation
    # ========================================================================
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Dataset Choice"):
            st.session_state.current_step = 'dataset_choice'
            st.rerun()

def scan_indiasat_datasets():
    """
    Scan IndiaSAT folder for available datasets
    Returns list of dataset information dictionaries
    """
    
    datasets = []
    
    # The main dataset file location (adjust path as needed)
    # IndiaSAT typically has data in: IndiaSAT/Datasets/
    
    dataset_path = Path("IndiaSAT/Datasets")
    
    if not dataset_path.exists():
        return datasets
    
    # Look for CSV or pickle files
    data_files = list(dataset_path.glob("*.csv")) + list(dataset_path.glob("*.pkl"))
    
    for file_path in data_files:
        try:
            # Try to load and inspect the file
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, nrows=5)  # Just peek at first few rows
                total_rows = sum(1 for _ in open(file_path)) - 1  # Count total rows
            else:
                df = pd.read_pickle(file_path)
                total_rows = len(df)
            
            # Extract information
            info = {
                'name': file_path.stem,
                'path': str(file_path),
                'region': 'India (Multiple Regions)',  # IndiaSAT covers multiple regions
                'years': '2013-2019',
                'satellite': 'Landsat 8',
                'samples': total_rows,
                'classes': len(df['label'].unique()) if 'label' in df.columns else 'Unknown',
                'size_mb': round(file_path.stat().st_size / (1024*1024), 2)
            }
            
            # Get class distribution
            if 'label' in df.columns:
                # Load full file for accurate distribution
                df_full = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_pickle(file_path)
                info['class_distribution'] = df_full['label'].value_counts().to_dict()
            
            datasets.append(info)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return datasets

def load_dataset(dataset_info):
    """
    Load the selected dataset into session state
    """
    
    st.info(f"Loading {dataset_info['name']}...")
    
    try:
        # Load the full dataset
        file_path = Path(dataset_info['path'])
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_pickle(file_path)
        
        st.success(f"✅ Loaded {len(df):,} samples!")
        
        # Show a preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show column information
        st.subheader("📋 Dataset Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Columns:**")
            for col in df.columns:
                st.text(f"• {col} ({df[col].dtype})")
        
        with col2:
            st.markdown("**Statistics:**")
            st.text(f"Total rows: {len(df):,}")
            st.text(f"Total columns: {len(df.columns)}")
            st.text(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for required columns
        required_cols = ['label']  # At minimum we need labels
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.info("This dataset might need preprocessing before use.")
            return
        
        # Class distribution visualization
        st.subheader("🏷️ Class Distribution")
        
        class_counts = df['label'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(class_counts)
        
        with col2:
            st.markdown("**Class Breakdown:**")
            for class_name, count in class_counts.items():
                percentage = (count / len(df)) * 100
                st.metric(
                    label=f"Class: {class_name}",
                    value=f"{count:,} samples",
                    delta=f"{percentage:.1f}%"
                )
        
        # Save to session state
        st.session_state.processed_data = df
        st.session_state.dataset_info = dataset_info
        
        st.success("✅ Dataset loaded successfully!")
        
        # Next step button
        st.markdown("---")
        
        if st.button("➡️ Continue to Data Splitting", type="primary", use_container_width=True):
            st.session_state.current_step = 'data_splitting'
            st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.exception(e)

"""
🎉 DONE!

This page handles loading the existing IndiaSAT dataset.

KEY FEATURES:
1. Checks if IndiaSAT exists
2. Scans for available datasets
3. Shows dataset information
4. Loads selected dataset
5. Displays preview and statistics

NEXT: Create the page for custom dataset creation!
"""
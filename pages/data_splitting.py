"""
PART 3F: DATA SPLITTING PAGE
=============================

Save this as: pages/data_splitting.py

This page splits data into training, validation, and test sets.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    """Main function to show data splitting page"""
    
    st.title("✂️ Split Data for Machine Learning")
    
    st.markdown("""
    Before training a model, we need to split our data into three parts:
    - **Training Set**: Used to train the model (learns from this)
    - **Validation Set**: Used to tune the model (checks performance during training)
    - **Test Set**: Final evaluation (see how well it really works)
    
    Think of it like studying for an exam:
    - Training = Study material you learn from
    - Validation = Practice tests to check progress
    - Test = The actual final exam
    """)
    
    # Check if data is loaded
    if st.session_state.processed_data is None:
        st.error("❌ No data loaded! Please go back and load a dataset first.")
        
        if st.button("← Back to Dataset Selection"):
            st.session_state.current_step = 'dataset_choice'
            st.rerun()
        return
    
    data = st.session_state.processed_data
    
    # ========================================================================
    # DATA OVERVIEW
    # ========================================================================
    
    st.subheader("📊 Data Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", f"{len(data):,}")
    
    with col2:
        st.metric("Features", len([col for col in data.columns if col != 'label']))
    
    with col3:
        if 'label' in data.columns:
            st.metric("Classes", data['label'].nunique())
    
    # Show class distribution
    if 'label' in data.columns:
        st.markdown("**Class Distribution:**")
        
        class_counts = data['label'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        class_counts.plot(kind='bar', ax=ax)
        plt.title("Number of Samples per Class")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Check for class imbalance
        max_class = class_counts.max()
        min_class = class_counts.min()
        imbalance_ratio = max_class / min_class
        
        if imbalance_ratio > 3:
            st.warning(f"""
            ⚠️ **Class Imbalance Detected!**
            
            Your dataset has unbalanced classes (ratio: {imbalance_ratio:.1f}:1).
            This means some classes have many more samples than others.
            
            **Why this matters:** The model might become biased toward common classes.
            
            **Solutions:** Use stratified splitting (recommended below) or class balancing techniques.
            """)
    
    # ========================================================================
    # SPLITTING STRATEGY
    # ========================================================================
    
    st.markdown("---")
    st.subheader("🎯 Splitting Strategy")
    
    split_strategy = st.radio(
        "Choose splitting method:",
        [
            "Random Split - Simple and fast",
            "Stratified Split - Maintains class balance (Recommended)",
            "Temporal Split - Split by time (for time-series)"
        ]
    )
    
    if "Random" in split_strategy:
        st.info("""
        **Random Split:** Randomly assigns samples to each set.
        - ✅ Simple and fast
        - ❌ May not preserve class distribution
        - Best for: Balanced datasets
        """)
        use_stratify = False
        use_temporal = False
    
    elif "Stratified" in split_strategy:
        st.info("""
        **Stratified Split:** Ensures each set has the same class distribution.
        - ✅ Maintains class balance
        - ✅ Better for imbalanced data
        - ❌ Slightly slower
        - Best for: Most use cases (RECOMMENDED)
        """)
        use_stratify = True
        use_temporal = False
    
    else:  # Temporal
        st.info("""
        **Temporal Split:** Splits by time order.
        - ✅ Realistic for time-series
        - ✅ Tests future prediction
        - ❌ May have different distributions
        - Best for: Time-series forecasting
        """)
        use_stratify = False
        use_temporal = True
    
    # ========================================================================
    # SPLIT RATIOS
    # ========================================================================
    
    st.markdown("---")
    st.subheader("📏 Split Ratios")
    
    st.markdown("**Define what percentage goes to each set:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_ratio = st.slider(
            "Training Set %",
            min_value=50,
            max_value=90,
            value=70,
            step=5,
            help="Typically 60-80%"
        )
    
    with col2:
        val_ratio = st.slider(
            "Validation Set %",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            help="Typically 10-20%"
        )
    
    with col3:
        test_ratio = 100 - train_ratio - val_ratio
        st.metric("Test Set %", f"{test_ratio}%", help="Remaining percentage")
    
    # Validate ratios
    if test_ratio < 5:
        st.error("❌ Test set is too small! Reduce training or validation set.")
        return
    
    # Show actual sample counts
    st.markdown("**Estimated sample counts:**")
    
    total_samples = len(data)
    train_count = int(total_samples * train_ratio / 100)
    val_count = int(total_samples * val_ratio / 100)
    test_count = total_samples - train_count - val_count
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training", f"{train_count:,} samples")
    with col2:
        st.metric("Validation", f"{val_count:,} samples")
    with col3:
        st.metric("Test", f"{test_count:,} samples")
    
    # ========================================================================
    # ADDITIONAL OPTIONS
    # ========================================================================
    
    st.markdown("---")
    st.subheader("⚙️ Additional Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        shuffle_data = st.checkbox(
            "Shuffle data before splitting",
            value=True,
            help="Randomize order before splitting"
        )
        
        set_seed = st.checkbox(
            "Use random seed (reproducible)",
            value=True,
            help="Get same split every time"
        )
        
        if set_seed:
            random_seed = st.number_input(
                "Random seed:",
                min_value=0,
                max_value=9999,
                value=42,
                help="Any number - use same for reproducibility"
            )
        else:
            random_seed = None
    
    with col2:
        save_splits = st.checkbox(
            "Save splits to separate files",
            value=True,
            help="Save as train.csv, val.csv, test.csv"
        )
        
        save_combined = st.checkbox(
            "Also save combined file with split labels",
            value=False,
            help="Single file with 'split' column"
        )
    
    # ========================================================================
    # PERFORM SPLIT
    # ========================================================================
    
    st.markdown("---")
    
    if st.button("✂️ Split Data Now!", type="primary", use_container_width=True):
        
        with st.spinner("Splitting data..."):
            
            try:
                # Prepare features and labels
                if 'label' in data.columns:
                    X = data.drop('label', axis=1)
                    y = data['label']
                else:
                    st.error("❌ No 'label' column found in data!")
                    return
                
                # Perform splitting
                if use_temporal:
                    # Temporal split - no shuffling
                    train_end = int(len(data) * train_ratio / 100)
                    val_end = train_end + int(len(data) * val_ratio / 100)
                    
                    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
                    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
                    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
                
                else:
                    # Random or stratified split
                    stratify_arg = y if use_stratify else None
                    
                    # First split: train + val vs test
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y,
                        test_size=test_ratio/100,
                        random_state=random_seed,
                        stratify=stratify_arg,
                        shuffle=shuffle_data
                    )
                    
                    # Second split: train vs val
                    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
                    stratify_temp = y_temp if use_stratify else None
                    
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp,
                        test_size=val_ratio_adjusted,
                        random_state=random_seed,
                        stratify=stratify_temp,
                        shuffle=shuffle_data
                    )
                
                # Combine back into DataFrames
                train_data = pd.concat([X_train, y_train], axis=1)
                val_data = pd.concat([X_val, y_val], axis=1)
                test_data = pd.concat([X_test, y_test], axis=1)
                
                # Save to session state
                st.session_state.train_data = train_data
                st.session_state.val_data = val_data
                st.session_state.test_data = test_data
                
                st.session_state.split_config = {
                    'strategy': split_strategy,
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'shuffle': shuffle_data,
                    'random_seed': random_seed,
                    'use_stratify': use_stratify
                }
                
                # Show success message
                st.success("✅ Data split successfully!")
                
                # Show split information
                show_split_info(train_data, val_data, test_data, y)
                
                # Save to files if requested
                if save_splits:
                    save_split_files(train_data, val_data, test_data, save_combined)
                
                # Show next step button
                st.markdown("---")
                if st.button("Next: Prepare for ML Model →", type="primary", use_container_width=True):
                    st.session_state.current_step = 'ml_preparation'
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error splitting data: {str(e)}")
                st.exception(e)

def show_split_info(train_data, val_data, test_data, original_labels):
    """Display information about the splits"""
    
    st.subheader("✅ Split Complete!")
    
    # Size comparison
    st.markdown("**Split Sizes:**")
    
    split_df = pd.DataFrame({
        'Set': ['Training', 'Validation', 'Test'],
        'Samples': [len(train_data), len(val_data), len(test_data)],
        'Percentage': [
            f"{len(train_data)/len(original_labels)*100:.1f}%",
            f"{len(val_data)/len(original_labels)*100:.1f}%",
            f"{len(test_data)/len(original_labels)*100:.1f}%"
        ]
    })
    
    st.dataframe(split_df, use_container_width=True, hide_index=True)
    
    # Class distribution comparison
    st.markdown("**Class Distribution in Each Set:**")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, data) in enumerate([('Training', train_data), ('Validation', val_data), ('Test', test_data)]):
        data['label'].value_counts().plot(kind='bar', ax=axes[idx])
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel('Count')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Check if distributions are similar
    train_dist = train_data['label'].value_counts(normalize=True).sort_index()
    val_dist = val_data['label'].value_counts(normalize=True).sort_index()
    test_dist = test_data['label'].value_counts(normalize=True).sort_index()
    
    max_diff = max(
        abs(train_dist - val_dist).max(),
        abs(train_dist - test_dist).max(),
        abs(val_dist - test_dist).max()
    )
    
    if max_diff < 0.05:
        st.success("✅ Class distributions are well-balanced across all sets!")
    elif max_diff < 0.10:
        st.info("ℹ️ Class distributions are reasonably balanced.")
    else:
        st.warning(f"⚠️ Class distributions vary across sets (max difference: {max_diff*100:.1f}%)")

def save_split_files(train_data, val_data, test_data, save_combined):
    """Save split data to files"""
    
    import os
    os.makedirs('data/splits', exist_ok=True)
    
    with st.spinner("Saving split files..."):
        # Save individual files
        train_data.to_csv('data/splits/train.csv', index=False)
        val_data.to_csv('data/splits/val.csv', index=False)
        test_data.to_csv('data/splits/test.csv', index=False)
        
        st.success("""
        ✅ Files saved:
        - `data/splits/train.csv`
        - `data/splits/val.csv`
        - `data/splits/test.csv`
        """)
        
        # Save combined file if requested
        if save_combined:
            combined = pd.concat([
                train_data.assign(split='train'),
                val_data.assign(split='val'),
                test_data.assign(split='test')
            ])
            combined.to_csv('data/splits/all_data_with_splits.csv', index=False)
            st.info("✅ Also saved: `data/splits/all_data_with_splits.csv`")

# ============================================================================
# NAVIGATION
# ============================================================================

    # Show navigation if split is done
    if st.session_state.train_data is not None:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("← Back"):
                # Determine where to go back based on dataset choice
                if st.session_state.dataset_choice == 'existing':
                    st.session_state.current_step = 'load_existing'
                else:
                    st.session_state.current_step = 'downloading'
                st.rerun()
        
        with col2:
            if st.button("Next: ML Preparation →", type="primary"):
                st.session_state.current_step = 'ml_preparation'
                st.rerun()

"""
🎉 DATA SPLITTING PAGE COMPLETE!

This page handles:
1. Data overview and class distribution
2. Multiple splitting strategies (random, stratified, temporal)
3. Customizable split ratios
4. Visualization of splits
5. Saving split data to files
6. Validation of split quality

Next: Create the ML preparation page!
"""
"""
PART 3G: ML PREPARATION & MODEL TRAINING PAGE
==============================================

Save this as: pages/ml_preparation.py

This prepares data for ML models and provides a template for training.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

def show():
    """Main function to show ML preparation page"""
    
    st.title("🤖 Prepare Data for Machine Learning")
    
    st.markdown("""
    Your data is split and ready! Now let's prepare it in the format needed 
    for machine learning models.
    """)
    
    # Check if splits exist
    if st.session_state.train_data is None:
        st.error("❌ No split data found! Please split your data first.")
        
        if st.button("← Back to Data Splitting"):
            st.session_state.current_step = 'data_splitting'
            st.rerun()
        return
    
    # Get split data
    train_data = st.session_state.train_data
    val_data = st.session_state.val_data
    test_data = st.session_state.test_data
    
    # ========================================================================
    # DATA SUMMARY
    # ========================================================================
    
    st.subheader("📊 Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Samples", f"{len(train_data):,}")
    with col2:
        st.metric("Validation Samples", f"{len(val_data):,}")
    with col3:
        st.metric("Test Samples", f"{len(test_data):,}")
    with col4:
        st.metric("Features", len(train_data.columns) - 1)  # Exclude label
    
    # ========================================================================
    # DATA FORMAT SELECTION
    # ========================================================================
    
    st.markdown("---")
    st.subheader("📦 Choose Data Format")
    
    format_choice = st.radio(
        "What format do you need for your ML model?",
        [
            "NumPy Arrays (.npy) - For PyTorch, TensorFlow",
            "CSV Files (.csv) - For scikit-learn, pandas",
            "HDF5 (.h5) - For large datasets",
            "PyTorch Tensors (.pt) - For PyTorch",
            "All Formats - Save everything"
        ]
    )
    
    # Show format details
    format_info = {
        "NumPy": "Efficient for deep learning frameworks. Stores features (X) and labels (y) separately.",
        "CSV": "Human-readable, works with any ML library. Best for smaller datasets.",
        "HDF5": "Compressed format for large datasets. Very efficient storage.",
        "PyTorch": "Native PyTorch format. Fastest loading for PyTorch models.",
        "All": "Saves in all formats for maximum flexibility."
    }
    
    selected_format = format_choice.split(" ")[0]
    st.info(f"ℹ️ {format_info[selected_format]}")
    
    # ========================================================================
    # FEATURE ENGINEERING OPTIONS
    # ========================================================================
    
    st.markdown("---")
    st.subheader("🔧 Feature Engineering (Optional)")
    
    with st.expander("Advanced Options"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            scale_features = st.checkbox(
                "Apply feature scaling",
                value=True,
                help="Normalize features to same scale"
            )
            
            if scale_features:
                scaler_type = st.selectbox(
                    "Scaler type:",
                    ["StandardScaler (z-score)", "MinMaxScaler (0-1)", "RobustScaler (outlier-resistant)"]
                )
            
            remove_correlated = st.checkbox(
                "Remove highly correlated features",
                value=False,
                help="Remove redundant features with correlation > 0.95"
            )
        
        with col2:
            pca_reduce = st.checkbox(
                "Apply PCA dimensionality reduction",
                value=False,
                help="Reduce number of features while keeping variance"
            )
            
            if pca_reduce:
                variance_kept = st.slider(
                    "Variance to keep (%):",
                    min_value=80,
                    max_value=99,
                    value=95
                )
            
            handle_missing = st.selectbox(
                "Handle missing values:",
                ["Keep as is", "Fill with mean", "Fill with median", "Fill with mode", "Remove rows"]
            )
    
    # ========================================================================
    # LABEL ENCODING
    # ========================================================================
    
    st.markdown("---")
    st.subheader("🏷️ Label Encoding")
    
    # Get unique labels
    unique_labels = sorted(train_data['label'].unique())
    
    st.markdown(f"**Found {len(unique_labels)} unique classes:**")
    
    # Check if labels are already numeric
    if all(isinstance(label, (int, float)) for label in unique_labels):
        st.success("✅ Labels are already numeric! No encoding needed.")
        label_mapping = {label: label for label in unique_labels}
    else:
        st.info("Labels are text - we'll convert them to numbers for ML models.")
        
        # Create label mapping
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Show mapping table
        mapping_df = pd.DataFrame({
            'Original Label': list(label_mapping.keys()),
            'Encoded Value': list(label_mapping.values())
        })
        
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        
        # Option to customize mapping
        if st.checkbox("Customize label mapping"):
            st.warning("⚠️ Advanced users only!")
            
            for label in unique_labels:
                new_value = st.number_input(
                    f"Value for '{label}':",
                    min_value=0,
                    max_value=len(unique_labels)-1,
                    value=label_mapping[label],
                    key=f"label_{label}"
                )
                label_mapping[label] = new_value
    
    # ========================================================================
    # PREVIEW PREPARED DATA
    # ========================================================================
    
    st.markdown("---")
    st.subheader("👀 Preview Prepared Data")
    
    # Prepare a sample
    sample_train = train_data.head(5).copy()
    
    # Apply label encoding to sample
    sample_train['label_encoded'] = sample_train['label'].map(label_mapping)
    
    st.markdown("**Sample of training data (first 5 rows):**")
    st.dataframe(sample_train, use_container_width=True)
    
    # ========================================================================
    # DATA PREPARATION & EXPORT
    # ========================================================================
    
    st.markdown("---")
    st.subheader("💾 Prepare & Export Data")
    
    if st.button("🚀 Prepare Data for ML!", type="primary", use_container_width=True):
        
        with st.spinner("Preparing data..."):
            
            try:
                # Create output directory
                Path('data/ml_ready').mkdir(parents=True, exist_ok=True)
                
                # Apply label encoding
                train_encoded = encode_labels(train_data, label_mapping)
                val_encoded = encode_labels(val_data, label_mapping)
                test_encoded = encode_labels(test_data, label_mapping)
                
                # Separate features and labels
                X_train = train_encoded.drop('label', axis=1).values
                y_train = train_encoded['label'].values
                
                X_val = val_encoded.drop('label', axis=1).values
                y_val = val_encoded['label'].values
                
                X_test = test_encoded.drop('label', axis=1).values
                y_test = test_encoded['label'].values
                
                # Apply feature scaling if requested
                if scale_features:
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                    
                    if "Standard" in scaler_type:
                        scaler = StandardScaler()
                    elif "MinMax" in scaler_type:
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()
                    
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)
                    X_test = scaler.transform(X_test)
                    
                    # Save scaler
                    import joblib
                    joblib.dump(scaler, 'data/ml_ready/scaler.pkl')
                    st.info("✅ Saved scaler to `data/ml_ready/scaler.pkl`")
                
                # Save in requested formats
                saved_files = []
                
                if "NumPy" in format_choice or "All" in format_choice:
                    save_numpy_format(X_train, y_train, X_val, y_val, X_test, y_test)
                    saved_files.append("NumPy arrays (.npy)")
                
                if "CSV" in format_choice or "All" in format_choice:
                    save_csv_format(train_encoded, val_encoded, test_encoded)
                    saved_files.append("CSV files (.csv)")
                
                if "HDF5" in format_choice or "All" in format_choice:
                    save_hdf5_format(X_train, y_train, X_val, y_val, X_test, y_test)
                    saved_files.append("HDF5 file (.h5)")
                
                if "PyTorch" in format_choice or "All" in format_choice:
                    save_pytorch_format(X_train, y_train, X_val, y_val, X_test, y_test)
                    saved_files.append("PyTorch tensors (.pt)")
                
                # Save metadata
                metadata = {
                    'num_features': X_train.shape[1],
                    'num_classes': len(unique_labels),
                    'label_mapping': label_mapping,
                    'feature_names': list(train_data.columns.drop('label')),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                    'scaled': scale_features,
                    'scaler_type': scaler_type if scale_features else None
                }
                
                with open('data/ml_ready/metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                st.success(f"""
                🎉 **Data Preparation Complete!**
                
                ✅ Saved formats: {', '.join(saved_files)}
                
                📁 Location: `data/ml_ready/`
                
                📋 Metadata saved to: `metadata.json`
                """)
                
                # Show file structure
                show_file_structure()
                
                # Show next steps
                show_next_steps(metadata)
                
            except Exception as e:
                st.error(f"❌ Error preparing data: {str(e)}")
                st.exception(e)

def encode_labels(data, mapping):
    """Encode labels using the provided mapping"""
    data_copy = data.copy()
    data_copy['label'] = data_copy['label'].map(mapping)
    return data_copy

def save_numpy_format(X_train, y_train, X_val, y_val, X_test, y_test):
    """Save data in NumPy format"""
    np.save('data/ml_ready/X_train.npy', X_train)
    np.save('data/ml_ready/y_train.npy', y_train)
    np.save('data/ml_ready/X_val.npy', X_val)
    np.save('data/ml_ready/y_val.npy', y_val)
    np.save('data/ml_ready/X_test.npy', X_test)
    np.save('data/ml_ready/y_test.npy', y_test)

def save_csv_format(train_data, val_data, test_data):
    """Save data in CSV format"""
    train_data.to_csv('data/ml_ready/train.csv', index=False)
    val_data.to_csv('data/ml_ready/val.csv', index=False)
    test_data.to_csv('data/ml_ready/test.csv', index=False)

def save_hdf5_format(X_train, y_train, X_val, y_val, X_test, y_test):
    """Save data in HDF5 format"""
    import h5py
    
    with h5py.File('data/ml_ready/data.h5', 'w') as f:
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_val', data=X_val)
        f.create_dataset('y_val', data=y_val)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)

def save_pytorch_format(X_train, y_train, X_val, y_val, X_test, y_test):
    """Save data in PyTorch format"""
    import torch
    
    torch.save({
        'X_train': torch.FloatTensor(X_train),
        'y_train': torch.LongTensor(y_train),
        'X_val': torch.FloatTensor(X_val),
        'y_val': torch.LongTensor(y_val),
        'X_test': torch.FloatTensor(X_test),
        'y_test': torch.LongTensor(y_test)
    }, 'data/ml_ready/data.pt')

def show_file_structure():
    """Show the created file structure"""
    
    st.markdown("---")
    st.subheader("📁 Created Files")
    
    st.code("""
data/ml_ready/
├── metadata.json          # Dataset information
├── scaler.pkl            # Feature scaler (if scaling applied)
│
├── X_train.npy           # Training features (NumPy)
├── y_train.npy           # Training labels (NumPy)
├── X_val.npy             # Validation features (NumPy)
├── y_val.npy             # Validation labels (NumPy)
├── X_test.npy            # Test features (NumPy)
├── y_test.npy            # Test labels (NumPy)
│
├── train.csv             # Training set (CSV)
├── val.csv               # Validation set (CSV)
├── test.csv              # Test set (CSV)
│
├── data.h5               # All data (HDF5)
└── data.pt               # All data (PyTorch)
    """, language="text")

def show_next_steps(metadata):
    """Show instructions for next steps"""
    
    st.markdown("---")
    st.subheader("🎯 Next Steps: Train Your Model")
    
    st.markdown("""
    ### Your data is ready! Here's how to use it:
    """)
    
    # Create tabs for different ML frameworks
    tab1, tab2, tab3 = st.tabs(["🔥 PyTorch", "🧮 scikit-learn", "🌊 TensorFlow"])
    
    with tab1:
        show_pytorch_example(metadata)
    
    with tab2:
        show_sklearn_example(metadata)
    
    with tab3:
        show_tensorflow_example(metadata)
    
    # Model training button
    st.markdown("---")
    
    if st.button("Go to Model Training →", type="primary", use_container_width=True):
        st.session_state.current_step = 'model_training'
        st.rerun()

def show_pytorch_example(metadata):
    """Show PyTorch code example"""
    
    st.markdown("**PyTorch Example:**")
    
    code = f"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Load data
data = torch.load('data/ml_ready/data.pt')

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define model
class LandCoverClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear({metadata['num_features']}, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, {metadata['num_classes']})
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model
model = LandCoverClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {{epoch+1}} complete")

# YOUR MODEL TRAINING CODE GOES HERE!
    """
    
    st.code(code, language="python")

def show_sklearn_example(metadata):
    """Show scikit-learn code example"""
    
    st.markdown("**scikit-learn Example:**")
    
    code = f"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
X_train = np.load('data/ml_ready/X_train.npy')
y_train = np.load('data/ml_ready/y_train.npy')
X_val = np.load('data/ml_ready/X_val.npy')
y_val = np.load('data/ml_ready/y_val.npy')

# Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_val, model.predict(X_val))

print(f"Training Accuracy: {{train_acc:.4f}}")
print(f"Validation Accuracy: {{val_acc:.4f}}")

# Detailed metrics
y_pred = model.predict(X_val)
print("\\nClassification Report:")
print(classification_report(y_val, y_pred))

# YOUR MODEL TRAINING CODE GOES HERE!
    """
    
    st.code(code, language="python")

def show_tensorflow_example(metadata):
    """Show TensorFlow code example"""
    
    st.markdown("**TensorFlow/Keras Example:**")
    
    code = f"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
X_train = np.load('data/ml_ready/X_train.npy')
y_train = np.load('data/ml_ready/y_train.npy')
X_val = np.load('data/ml_ready/X_val.npy')
y_val = np.load('data/ml_ready/y_val.npy')

# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=({metadata['num_features']},)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense({metadata['num_classes']}, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {{test_acc:.4f}}")

# YOUR MODEL TRAINING CODE GOES HERE!
    """
    
    st.code(code, language="python")

# ============================================================================
# NAVIGATION
# ============================================================================

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Data Splitting"):
            st.session_state.current_step = 'data_splitting'
            st.rerun()

"""
🎉 ML PREPARATION PAGE COMPLETE!

This page handles:
1. Data format conversion (NumPy, CSV, HDF5, PyTorch)
2. Feature scaling options
3. Label encoding
4. Data export in multiple formats
5. Code examples for different ML frameworks

The user can now take this prepared data and train their models!
"""
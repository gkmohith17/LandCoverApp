# 🛰️ Land Cover Classification Application - Complete Guide

## 📚 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Running the Application](#running-the-application)
5. [Step-by-Step Usage](#step-by-step-usage)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

---

## 🌟 Overview

This is a complete end-to-end application for creating land cover classification datasets using satellite imagery. It handles everything from data download to preparing data for machine learning models.

### What It Does:

- 🛰️ Downloads satellite imagery from Google Earth Engine
- 🗺️ Lets you select custom regions anywhere in the world
- ⚙️ Preprocesses data (cloud masking, spectral indices)
- ✂️ Splits data into train/validation/test sets
- 💾 Exports in multiple formats (NumPy, CSV, PyTorch, etc.)
- 🤖 Prepares data ready for ML model training

---

## 📁 Project Structure

After setup, your project will look like this:

```
landcover_app/
│
├── app.py                      # Main application entry point
│
├── pages/                      # Individual page modules
│   ├── __init__.py
│   ├── load_existing_dataset.py
│   ├── region_selection.py
│   ├── timeline_config.py
│   ├── preprocessing.py
│   ├── data_download.py
│   ├── data_splitting.py
│   ├── ml_preparation.py
│   └── model_training.py
│
├── utils/                      # Helper functions
│   ├── __init__.py
│   └── helpers.py
│
├── data/                       # All data files
│   ├── raw/                    # Downloaded raw data
│   ├── processed/              # Processed data
│   ├── splits/                 # Train/val/test splits
│   └── ml_ready/               # Final ML-ready data
│
├── models/                     # Saved ML models
│
├── temp/                       # Temporary files
│
├── IndiaSAT/                   # (Optional) IndiaSAT dataset
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Installation

### Prerequisites:

- **Python 3.8+** (Download from python.org)
- **pip** (comes with Python)
- **Google Account** (for Google Earth Engine)

### Step 1: Create Project Folder

```bash
# Windows (Command Prompt)
mkdir landcover_app
cd landcover_app

# Mac/Linux (Terminal)
mkdir landcover_app && cd landcover_app
```

### Step 2: Create requirements.txt

Create a file named `requirements.txt` with this content:

```txt
streamlit==1.31.0
earthengine-api==0.1.398
rasterio==1.3.9
geopandas==0.14.3
shapely==2.0.3
pyproj==3.6.1
folium==0.15.1
streamlit-folium==0.15.1
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.2
matplotlib==3.7.1
seaborn==0.13.2
plotly==5.18.0
tqdm==4.66.1
Pillow==10.2.0
h5py==3.9.0
joblib==1.3.2
```

### Step 3: Install Packages

```bash
pip install -r requirements.txt
```

This will take 5-10 minutes. Wait for completion.

### Step 4: Setup Google Earth Engine

**4.1: Register for GEE**

1. Go to: https://earthengine.google.com/signup/
2. Sign in with Google account
3. Fill out registration form
4. Wait for approval (usually instant)

**4.2: Authenticate**

```bash
earthengine authenticate
```

- Browser will open
- Sign in with same Google account
- Click "Allow"
- Close browser when done

**4.3: Test Authentication**

```python
python -c "import ee; ee.Initialize(); print('✅ GEE Working!')"
```

If you see "✅ GEE Working!", you're ready!

### Step 5: Create Folder Structure

```bash
# Windows
mkdir data data\raw data\processed data\splits data\ml_ready models utils pages temp

# Mac/Linux
mkdir -p data/{raw,processed,splits,ml_ready} models utils pages temp
```

### Step 6: Create Empty __init__.py Files

```bash
# Windows
type nul > utils\__init__.py
type nul > pages\__init__.py

# Mac/Linux
touch utils/__init__.py pages/__init__.py
```

---

## 🏃 Running the Application

### Quick Start:

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

### If Browser Doesn't Open:

Manually open your browser and go to: `http://localhost:8501`

### Stopping the App:

Press `Ctrl+C` in the terminal

---

## 📖 Step-by-Step Usage

### Option A: Using Existing IndiaSAT Dataset

**Perfect for:** Learning, quick experiments, testing

**Steps:**

1. **Download IndiaSAT:**
   ```bash
   git clone https://github.com/ChahatBansal8060/IndiaSAT.git
   ```
   
   OR manually download from GitHub and extract to project folder

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **In the app:**
   - Click "Use Existing Dataset"
   - Select a dataset from the list
   - Click "Load Dataset"
   - Review the data preview
   - Click "Continue to Data Splitting"

4. **Split the data:**
   - Choose split strategy (Stratified recommended)
   - Set ratios (70/15/15 default is good)
   - Click "Split Data Now!"
   - Review split quality

5. **Prepare for ML:**
   - Choose data format (NumPy for deep learning)
   - Enable feature scaling (recommended)
   - Click "Prepare Data for ML!"
   - Your data is ready!

**Time:** 5-10 minutes

---

### Option B: Creating Custom Dataset

**Perfect for:** Research, specific regions, custom requirements

**Steps:**

1. **Start the app:**
   ```bash
   streamlit run app.py
   ```

2. **Select region:**
   - Click "Create Custom Dataset"
   - Choose region selection method:
     - **Draw on map** (easiest)
     - **Enter coordinates**
     - **Upload shapefile**
     - **Select Indian state/district**
   - Draw/select your region
   - Click "Save This Region"

3. **Configure timeline:**
   - Choose satellite (Landsat 8 recommended)
   - Set date range (e.g., 2018-2023)
   - Select temporal aggregation (Yearly recommended)
   - Set cloud threshold (20% default)
   - Review estimated data size
   - Click "Next"

4. **Configure preprocessing:**
   - Select spectral bands (use defaults)
   - Choose spectral indices (NDVI, NDWI, NDBI recommended)
   - Select normalization (Min-Max for deep learning)
   - Click "Start Download!"

5. **Download data:**
   - Wait while data downloads (5-15 minutes)
   - Monitor progress bar
   - Check detailed log if needed
   - Wait for "Download Complete!" message

6. **Split the data:**
   - Same as Option A steps 4-5

**Time:** 20-30 minutes (depending on region size and internet speed)

---

## 🎓 Understanding the Output

### After Data Splitting:

Your `data/splits/` folder will contain:

- `train.csv` - Training set
- `val.csv` - Validation set  
- `test.csv` - Test set

### After ML Preparation:

Your `data/ml_ready/` folder will contain:

**NumPy Format:**
- `X_train.npy` - Training features
- `y_train.npy` - Training labels
- `X_val.npy` - Validation features
- `y_val.npy` - Validation labels
- `X_test.npy` - Test features
- `y_test.npy` - Test labels

**Other Formats:**
- `data.h5` - HDF5 format (if selected)
- `data.pt` - PyTorch format (if selected)
- `train.csv`, `val.csv`, `test.csv` - CSV format

**Metadata:**
- `metadata.json` - Dataset information
- `scaler.pkl` - Feature scaler (if scaling applied)

---

## 🤖 Training Your Model

### Example 1: PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Load data
data = torch.load('data/ml_ready/data.pt')
X_train, y_train = data['X_train'], data['y_train']

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define model
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Train model
model = Classifier(X_train.shape[1], len(torch.unique(y_train)))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# YOUR TRAINING CODE HERE!
```

### Example 2: scikit-learn

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
X_train = np.load('data/ml_ready/X_train.npy')
y_train = np.load('data/ml_ready/y_train.npy')
X_val = np.load('data/ml_ready/X_val.npy')
y_val = np.load('data/ml_ready/y_val.npy')

# Train model
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
val_acc = accuracy_score(y_val, model.predict(X_val))
print(f"Validation Accuracy: {val_acc:.4f}")

# YOUR TRAINING CODE HERE!
```

---

## 🔧 Troubleshooting

### Common Issues:

**1. "pip is not recognized"**
- Solution: Reinstall Python and check "Add to PATH" during installation

**2. "earthengine authenticate" fails**
- Solution: Make sure you registered at https://earthengine.google.com/signup/
- Check approval email
- Try running: `earthengine authenticate --force`

**3. "No module named 'streamlit'"**
- Solution: Run `pip install -r requirements.txt` again
- Make sure you're in the correct folder

**4. App shows "Import Error: No module named 'pages'"**
- Solution: Make sure you created the `pages/__init__.py` file
- Check folder structure matches the guide

**5. Download is very slow**
- Solution: This is normal for large regions
- Try: Smaller region, shorter time period, yearly aggregation

**6. "Google Drive API not authenticated"**
- Solution: For now, files are saved to Google Drive but need manual download
- Check your Google Drive folder "LandCover_Data"

**7. Out of memory errors**
- Solution: Region too large
- Try: Split region into smaller tiles, reduce time period

**8. Windows: "Cannot create folder"**
- Solution: Run terminal as Administrator

---

## ⚙️ Advanced Configuration

### Customizing Satellite Collections:

Edit `pages/data_download.py`, function `load_satellite_collection()`:

```python
collections = {
    'Landsat 7': 'LANDSAT/LE07/C02/T1_L2',
    'Landsat 8': 'LANDSAT/LC08/C02/T1_L2',
    'Sentinel-2': 'COPERNICUS/S2_SR_HARMONIZED',
    # Add more satellites here!
}
```

### Adding Custom Spectral Indices:

Edit `pages/preprocessing.py`, add to `indices_info`:

```python
"CUSTOM_INDEX": {
    "name": "My Custom Index",
    "formula": "(Band1 - Band2) / (Band1 + Band2)",
    "range": "-1 to +1",
    "interpretation": "What it shows",
    "requires": ["Band1", "Band2"],
    "recommended": False
}
```

### Changing Split Ratios:

In `pages/data_splitting.py`, modify default values:

```python
train_ratio = st.slider("Training Set %", value=70)  # Change 70
val_ratio = st.slider("Validation Set %", value=15)   # Change 15
```

---

## 📚 Additional Resources

### Google Earth Engine:
- Documentation: https://developers.google.com/earth-engine
- Dataset Catalog: https://developers.google.com/earth-engine/datasets

### IndiaSAT Dataset:
- GitHub: https://github.com/ChahatBansal8060/IndiaSAT
- Paper: [Link to research paper if available]

### Streamlit:
- Documentation: https://docs.streamlit.io
- Community: https://discuss.streamlit.io

### Machine Learning:
- PyTorch: https://pytorch.org/tutorials/
- scikit-learn: https://scikit-learn.org/stable/
- TensorFlow: https://www.tensorflow.org/tutorials

---

## 📝 Notes

### Important Limits:

- **GEE Free Tier:**
  - Unlimited API calls (rate limited)
  - Max 10 concurrent exports
  - 2GB memory per query

- **Recommended Region Sizes:**
  - Small: < 1,000 km² (fast)
  - Medium: 1,000-10,000 km² (moderate)
  - Large: 10,000-50,000 km² (slow)
  - Very Large: > 50,000 km² (may fail, split into tiles)

### Best Practices:

1. **Start Small:** Test with small region first
2. **Use Stratified Split:** Maintains class balance
3. **Enable Cloud Filtering:** Removes bad images
4. **Save Configurations:** Use metadata.json for tracking
5. **Validate Results:** Check split distributions
6. **Regular Backups:** Copy data/ml_ready folder

---

## 🆘 Getting Help

### If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review error messages in the app
3. Check terminal output for detailed logs
4. Verify file structure matches guide
5. Ensure all dependencies installed correctly

### For bugs or questions:

- Create issue on GitHub (if available)
- Check Streamlit documentation
- Review Google Earth Engine docs

---

## 🎉 Success Checklist

Before training your model, make sure:

- [ ] App runs without errors
- [ ] Data successfully downloaded/loaded
- [ ] Data split into train/val/test
- [ ] Split distributions look balanced
- [ ] Files exist in `data/ml_ready/`
- [ ] `metadata.json` created
- [ ] Can load data in Python/PyTorch/etc.

If all checked, you're ready to train your land cover classification model!

---

## 📄 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

## 🙏 Acknowledgments

- Google Earth Engine team
- IndiaSAT dataset creators
- Streamlit community

---

**Happy classifying! 🛰️🌍**
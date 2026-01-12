"""
PART 5: AUTOMATIC SETUP SCRIPT
===============================

Save this as: setup_project.py

Run this script to automatically create all the necessary files and folders!

HOW TO USE:
1. Save this file as 'setup_project.py'
2. Run: python setup_project.py
3. Follow the on-screen instructions
"""

import os
import sys
from pathlib import Path

def create_folder_structure():
    """Create all necessary folders"""
    print("📁 Creating folder structure...")
    
    folders = [
        'data/raw',
        'data/processed',
        'data/splits',
        'data/ml_ready',
        'models',
        'utils',
        'pages',
        'temp'
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {folder}/")
    
    print("✅ Folder structure complete!\n")

def create_init_files():
    """Create __init__.py files for Python packages"""
    print("📝 Creating __init__.py files...")
    
    init_files = [
        'utils/__init__.py',
        'pages/__init__.py'
    ]
    
    for file in init_files:
        Path(file).touch()
        print(f"  ✅ Created: {file}")
    
    print("✅ Init files complete!\n")

def create_requirements_txt():
    """Create requirements.txt file"""
    print("📋 Creating requirements.txt...")
    
    requirements = """# Web Framework
streamlit==1.31.0

# Google Earth Engine
earthengine-api==0.1.398

# Geospatial Processing
rasterio==1.3.9
geopandas==0.14.3
shapely==2.0.3
pyproj==3.6.1

# Interactive Maps
folium==0.15.1
streamlit-folium==0.15.1

# Data Science
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.2

# Visualization
matplotlib==3.7.1
seaborn==0.13.2
plotly==5.18.0

# Utilities
tqdm==4.66.1
Pillow==10.2.0
h5py==3.9.0
joblib==1.3.2
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("  ✅ Created: requirements.txt")
    print("✅ Requirements file complete!\n")

def create_helpers_file():
    """Create utils/helpers.py"""
    print("🔧 Creating helper functions...")
    
    helpers_code = '''"""
Helper Functions
================

Common utility functions used throughout the application.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def create_folder(path):
    """Create a folder if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_config(config, filepath):
    """Save configuration to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filepath):
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_file_size_mb(filepath):
    """Get file size in megabytes"""
    size_bytes = os.path.getsize(filepath)
    return round(size_bytes / (1024 * 1024), 2)

def print_dataset_info(data):
    """Print dataset information"""
    print("\\n" + "="*50)
    print("📊 DATASET INFORMATION")
    print("="*50)
    print(f"Total samples: {len(data):,}")
    if 'label' in data.columns:
        print(f"\\nClass distribution:")
        print(data['label'].value_counts())
    print("="*50 + "\\n")
'''

    with open('utils/helpers.py', 'w', encoding='utf-8') as f:
        f.write(helpers_code)
    
    print("  ✅ Created: utils/helpers.py")
    print("✅ Helper functions complete!\n")

def create_gitignore():
    """Create .gitignore file"""
    print("🚫 Creating .gitignore...")
    
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Data files
data/
*.csv
*.npy
*.h5
*.pt
*.pkl
*.tif
*.tiff

# Temporary files
temp/
*.tmp

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Model files
models/*.h5
models/*.pt
models/*.pkl

# Logs
*.log
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore)
    
    print("  ✅ Created: .gitignore")
    print("✅ Gitignore complete!\n")

def check_python_version():
    """Check if Python version is adequate"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ Python 3.8+ required!")
        print(f"  Your version: Python {version.major}.{version.minor}.{version.micro}")
        print("  Please upgrade Python: https://www.python.org/downloads/")
        return False
    
    print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} detected")
    print("✅ Python version OK!\n")
    return True

def install_packages():
    """Offer to install packages"""
    print("📦 Ready to install packages...\n")
    
    response = input("Do you want to install packages now? (y/n): ").lower()
    
    if response == 'y':
        print("\n⏳ Installing packages... This may take 5-10 minutes.\n")
        
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        
        if result.returncode == 0:
            print("\n✅ Packages installed successfully!")
        else:
            print("\n❌ Package installation failed. Please run manually:")
            print("   pip install -r requirements.txt")
    else:
        print("\n⏭️ Skipping package installation.")
        print("   Run later: pip install -r requirements.txt")

def setup_gee():
    """Guide user through GEE setup"""
    print("\n" + "="*60)
    print("🌍 GOOGLE EARTH ENGINE SETUP")
    print("="*60)
    print("""
Google Earth Engine is required for downloading satellite data.

STEPS TO SETUP:

1. Register for Google Earth Engine (if you haven't):
   → Go to: https://earthengine.google.com/signup/
   → Sign in with your Google account
   → Fill out the registration form
   → Wait for approval email (usually instant)

2. Authenticate on your computer:
   → Run this command: earthengine authenticate
   → Browser will open
   → Sign in and click 'Allow'
   → Close browser when done

3. Test authentication:
   → Run: python -c "import ee; ee.Initialize(); print('GEE Works!')"

Do you want instructions to authenticate now? (y/n): """)
    
    response = input().lower()
    
    if response == 'y':
        print("\n📝 AUTHENTICATION STEPS:")
        print("1. Open a NEW terminal/command prompt")
        print("2. Run: earthengine authenticate")
        print("3. Follow browser instructions")
        print("4. Come back here when done\n")
        
        input("Press ENTER when authentication is complete...")
        
        # Test authentication
        try:
            import ee
            ee.Initialize()
            print("✅ Google Earth Engine authenticated successfully!")
        except:
            print("⚠️  Could not verify GEE authentication.")
            print("   You can try again later with: earthengine authenticate")
    else:
        print("\n⏭️ Skipping GEE setup for now.")
        print("   You'll need to authenticate before downloading data.")

def create_readme():
    """Create a simple README"""
    print("\n📄 Creating README...")
    
    readme = """# Land Cover Classification Application

## Quick Start

1. Install packages:
   ```
   pip install -r requirements.txt
   ```

2. Authenticate Google Earth Engine:
   ```
   earthengine authenticate
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

4. Open browser to: http://localhost:8501

## Need Help?

See the comprehensive README for detailed instructions!

Happy classifying! 🛰️
"""

    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print("  ✅ Created: README.md")

def show_next_steps():
    """Show what to do next"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("""
Your project structure is ready!

NEXT STEPS:

1. Copy the code artifacts:
   → Save app.py (main application)
   → Save all page modules in pages/ folder
   → Each artifact has clear filename instructions

2. Install packages (if not done):
   → Run: pip install -r requirements.txt

3. Setup Google Earth Engine (if not done):
   → Run: earthengine authenticate

4. (Optional) Download IndiaSAT dataset:
   → Run: git clone https://github.com/ChahatBansal8060/IndiaSAT.git

5. Run the application:
   → Run: streamlit run app.py
   → Browser opens automatically!

PROJECT STRUCTURE:
""")
    
    print("""
landcover_app/
├── app.py                 ← NEED TO CREATE (from artifacts)
├── requirements.txt       ✅ Created
├── README.md             ✅ Created
├── .gitignore            ✅ Created
│
├── pages/                ✅ Created
│   ├── __init__.py       ✅ Created
│   └── [page modules]    ← NEED TO CREATE (from artifacts)
│
├── utils/                ✅ Created
│   ├── __init__.py       ✅ Created
│   └── helpers.py        ✅ Created
│
└── data/                 ✅ Created
    ├── raw/              ✅ Created
    ├── processed/        ✅ Created
    ├── splits/           ✅ Created
    └── ml_ready/         ✅ Created

FILES TO CREATE FROM ARTIFACTS:
1. app.py
2. pages/load_existing_dataset.py
3. pages/region_selection.py
4. pages/timeline_config.py
5. pages/preprocessing.py
6. pages/data_download.py
7. pages/data_splitting.py
8. pages/ml_preparation.py
9. pages/model_training.py (optional - for your ML model)

Each artifact in the chat has clear save instructions at the top!

Good luck! 🚀
""")

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("🛰️  LAND COVER CLASSIFICATION APP - AUTO SETUP")
    print("="*60)
    print("\nThis script will create all necessary files and folders.\n")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create structure
    create_folder_structure()
    create_init_files()
    create_requirements_txt()
    create_helpers_file()
    create_gitignore()
    create_readme()
    
    # Offer to install packages
    install_packages()
    
    # GEE setup guide
    setup_gee()
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()
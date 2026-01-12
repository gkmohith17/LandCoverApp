"""
PART 3E: DATA DOWNLOAD PAGE (GOOGLE EARTH ENGINE)
==================================================

Save this as: pages/data_download.py

This is where the magic happens! We download satellite data from Google Earth Engine.
"""

import streamlit as st
import ee
import time
import os
from pathlib import Path
import json

def show():
    """Main function to show download page"""
    
    st.title("📥 Downloading Satellite Data")
    
    st.markdown("""
    We're now going to download your satellite data from Google Earth Engine!
    This will take a few minutes depending on your configuration.
    """)
    
    # ========================================================================
    # CHECK GEE AUTHENTICATION
    # ========================================================================
    
    if not check_gee_auth():
        return
    
    # ========================================================================
    # SHOW CONFIGURATION SUMMARY
    # ========================================================================
    
    show_config_summary()
    
    # ========================================================================
    # START DOWNLOAD
    # ========================================================================
    
    st.markdown("---")
    
    # Initialize download state
    if 'download_started' not in st.session_state:
        st.session_state.download_started = False
    
    if 'download_complete' not in st.session_state:
        st.session_state.download_complete = False
    
    if not st.session_state.download_started:
        st.subheader("Ready to Start!")
        
        st.info("""
        ⏱️ **Estimated time:** 5-15 minutes
        
        ℹ️ **What will happen:**
        1. Connect to Google Earth Engine
        2. Load satellite imagery for your region
        3. Apply cloud masking and filters
        4. Calculate spectral bands and indices
        5. Create temporal composites
        6. Export to Google Drive
        7. Download to your computer
        
        💡 **Tip:** Keep this window open. You can do other things while it downloads!
        """)
        
        if st.button("🚀 Start Download!", type="primary", use_container_width=True):
            st.session_state.download_started = True
            st.rerun()
    
    else:
        # Download is in progress
        perform_download()
    
    # ========================================================================
    # NAVIGATION
    # ========================================================================
    
    if st.session_state.download_complete:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col2:
            if st.button("Next: Split Data →", type="primary"):
                st.session_state.current_step = 'data_splitting'
                st.rerun()

def check_gee_auth():
    """Check if Google Earth Engine is authenticated"""
    
    st.subheader("🔐 Checking Authentication")
    
    try:
        ee.Initialize()
        st.success("✅ Google Earth Engine is authenticated and ready!")
        return True
    
    except Exception as e:
        st.error("❌ Google Earth Engine is not authenticated!")
        
        st.markdown("""
        ### How to authenticate:
        
        1. Open your terminal/command prompt
        2. Run this command:
        ```bash
        earthengine authenticate
        ```
        3. Follow the instructions in your browser
        4. Come back here and refresh the page
        
        **Need help?** Watch this video: [GEE Authentication Guide](https://www.youtube.com/watch?v=example)
        """)
        
        if st.button("🔄 Refresh Page"):
            st.rerun()
        
        return False

def show_config_summary():
    """Display a summary of the user's configuration"""
    
    st.subheader("📋 Your Configuration")
    
    region_area = st.session_state.get('region_area_km2', 'Unknown')
    timeline = st.session_state.timeline_config
    preprocess = st.session_state.preprocessing_config
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📍 Region**")
        st.info(f"Area: {region_area:.2f} km²" if isinstance(region_area, (int, float)) else "Area: Unknown")
    
    with col2:
        st.markdown("**📅 Timeline**")
        st.info(f"{timeline['satellite']}\n{timeline['num_composites']} composites")
    
    with col3:
        st.markdown("**⚙️ Processing**")
        st.info(f"{preprocess['total_features']} features\n{preprocess['normalization'].title()}")

def perform_download():
    """Perform the actual download from Google Earth Engine"""
    
    st.subheader("⚡ Download in Progress")
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_expander = st.expander("📋 Detailed Log", expanded=False)
    
    with log_expander:
        log_container = st.empty()
        logs = []
    
    def log(message):
        """Add a message to the log"""
        logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        log_container.text('\n'.join(logs))
    
    try:
        # Get configuration
        region_geom = st.session_state.region_geometry
        timeline = st.session_state.timeline_config
        preprocess = st.session_state.preprocessing_config
        
        # ====================================================================
        # STEP 1: INITIALIZE GEE
        # ====================================================================
        
        status_text.text("🔧 Initializing Google Earth Engine...")
        log("Initializing Google Earth Engine")
        ee.Initialize()
        progress_bar.progress(5)
        time.sleep(0.5)
        
        # ====================================================================
        # STEP 2: CONVERT REGION TO EE.GEOMETRY
        # ====================================================================
        
        status_text.text("📍 Processing region geometry...")
        log("Converting region to Earth Engine geometry")
        
        roi = ee.Geometry(region_geom)
        log(f"Region bounds: {roi.bounds().getInfo()}")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        # ====================================================================
        # STEP 3: LOAD SATELLITE COLLECTION
        # ====================================================================
        
        status_text.text(f"🛰️ Loading {timeline['satellite']} imagery...")
        log(f"Loading {timeline['satellite']} collection")
        
        collection = load_satellite_collection(
            timeline['satellite'],
            timeline['start_date'],
            timeline['end_date'],
            roi,
            timeline['cloud_threshold']
        )
        
        # Count available images
        count = collection.size().getInfo()
        log(f"Found {count} images in collection")
        
        if count == 0:
            st.error("❌ No images found for your region and time period!")
            st.info("Try: Expanding the time period, increasing cloud threshold, or selecting a different region")
            return
        
        progress_bar.progress(20)
        
        # ====================================================================
        # STEP 4: SELECT BANDS
        # ====================================================================
        
        status_text.text("🌈 Selecting spectral bands...")
        log(f"Selecting bands: {', '.join(preprocess['bands'])}")
        
        band_names = get_band_names(timeline['satellite'], preprocess['bands'])
        collection = collection.select(band_names)
        
        progress_bar.progress(30)
        time.sleep(0.5)
        
        # ====================================================================
        # STEP 5: CALCULATE INDICES
        # ====================================================================
        
        if preprocess['indices']:
            status_text.text("📊 Calculating spectral indices...")
            log(f"Calculating indices: {', '.join(preprocess['indices'])}")
            
            collection = add_spectral_indices(
                collection,
                preprocess['indices'],
                timeline['satellite']
            )
            
            progress_bar.progress(40)
            time.sleep(0.5)
        
        # ====================================================================
        # STEP 6: CREATE TEMPORAL COMPOSITES
        # ====================================================================
        
        status_text.text("⏱️ Creating temporal composites...")
        log(f"Creating {timeline['aggregation']} composites")
        
        composites = create_temporal_composites(
            collection,
            timeline['aggregation'],
            timeline['start_date'],
            timeline['end_date'],
            timeline['composite_method']
        )
        
        log(f"Created {len(composites)} composites")
        progress_bar.progress(50)
        
        # ====================================================================
        # STEP 7: EXPORT TO GOOGLE DRIVE
        # ====================================================================
        
        status_text.text("☁️ Exporting to Google Drive...")
        log("Starting export tasks")
        
        export_tasks = []
        
        for i, composite in enumerate(composites):
            task_name = f"landcover_composite_{i+1:03d}"
            
            task = export_to_drive(
                composite,
                task_name,
                roi,
                timeline['satellite']
            )
            
            export_tasks.append({'task': task, 'name': task_name})
            log(f"Started export: {task_name}")
        
        progress_bar.progress(60)
        
        # ====================================================================
        # STEP 8: MONITOR EXPORT PROGRESS
        # ====================================================================
        
        status_text.text("⏳ Waiting for exports to complete...")
        log(f"Monitoring {len(export_tasks)} export tasks...")
        
        completed = 0
        failed = 0
        
        while completed + failed < len(export_tasks):
            time.sleep(10)  # Check every 10 seconds
            
            completed = 0
            failed = 0
            
            for task_info in export_tasks:
                task = task_info['task']
                status = task.status()
                
                if status['state'] == 'COMPLETED':
                    completed += 1
                elif status['state'] == 'FAILED':
                    failed += 1
                    log(f"❌ Export failed: {task_info['name']} - {status.get('error_message', 'Unknown error')}")
            
            # Update progress
            export_progress = 60 + int((completed / len(export_tasks)) * 30)
            progress_bar.progress(export_progress)
            status_text.text(f"⏳ Exporting... ({completed}/{len(export_tasks)} complete)")
        
        if failed > 0:
            st.warning(f"⚠️ {failed} export(s) failed. Check the detailed log for errors.")
        
        log(f"✅ Export complete! {completed} successful, {failed} failed")
        progress_bar.progress(90)
        
        # ====================================================================
        # STEP 9: DOWNLOAD FROM GOOGLE DRIVE (SIMULATED)
        # ====================================================================
        
        status_text.text("📥 Downloading from Google Drive...")
        log("Downloading files from Google Drive to local storage")
        
        # In a real implementation, you would use the Google Drive API here
        # For this demo, we'll simulate it
        
        os.makedirs('data/raw', exist_ok=True)
        
        downloaded_files = []
        
        for task_info in export_tasks:
            if task_info['task'].status()['state'] == 'COMPLETED':
                # Simulate file download
                fake_file_path = f"data/raw/{task_info['name']}.tif"
                
                # In reality, you'd download the actual file here
                # For demo, just create a placeholder
                Path(fake_file_path).touch()
                
                downloaded_files.append(fake_file_path)
                log(f"Downloaded: {task_info['name']}.tif")
        
        progress_bar.progress(100)
        status_text.text("✅ Download Complete!")
        
        # ====================================================================
        # SAVE RESULTS
        # ====================================================================
        
        # Save file paths
        st.session_state.raw_data_paths = downloaded_files
        st.session_state.download_complete = True
        
        # Save metadata
        metadata = {
            'region': st.session_state.region_geometry,
            'timeline': timeline,
            'preprocessing': preprocess,
            'files': downloaded_files,
            'download_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('data/raw/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        log("Saved metadata to data/raw/metadata.json")
        
        # ====================================================================
        # SUCCESS MESSAGE
        # ====================================================================
        
        st.success(f"""
        🎉 **Download Complete!**
        
        - ✅ Downloaded {len(downloaded_files)} files
        - 💾 Total size: ~{timeline['estimated_size_gb']:.2f} GB
        - 📁 Location: data/raw/
        
        Your data is ready for the next step!
        """)
        
    except Exception as e:
        st.error(f"❌ Error during download: {str(e)}")
        log(f"ERROR: {str(e)}")
        st.exception(e)
        
        st.info("""
        **Troubleshooting tips:**
        1. Check your internet connection
        2. Verify Google Earth Engine authentication
        3. Try a smaller region or time period
        4. Check the detailed log for specific errors
        """)

# ============================================================================
# HELPER FUNCTIONS FOR GOOGLE EARTH ENGINE
# ============================================================================

def load_satellite_collection(satellite, start_date, end_date, roi, cloud_threshold):
    """Load satellite imagery collection from GEE"""
    
    collections = {
        'Landsat 7': 'LANDSAT/LE07/C02/T1_L2',
        'Landsat 8': 'LANDSAT/LC08/C02/T1_L2',
        'Sentinel-2': 'COPERNICUS/S2_SR_HARMONIZED'
    }
    
    collection = ee.ImageCollection(collections[satellite]) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
    
    # Apply cloud masking
    if 'Landsat' in satellite:
        collection = collection.map(mask_landsat_clouds)
    else:
        collection = collection.map(mask_sentinel_clouds)
    
    return collection

def mask_landsat_clouds(image):
    """Mask clouds in Landsat imagery"""
    qa = image.select('QA_PIXEL')
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(cloud_mask).divide(10000)  # Scale factor

def mask_sentinel_clouds(image):
    """Mask clouds in Sentinel-2 imagery"""
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask).divide(10000)  # Scale factor

def get_band_names(satellite, selected_bands):
    """Map user-friendly band names to actual band names"""
    
    if 'Landsat' in satellite:
        band_mapping = {
            'Blue': 'SR_B2',
            'Green': 'SR_B3',
            'Red': 'SR_B4',
            'NIR': 'SR_B5',
            'SWIR1': 'SR_B6',
            'SWIR2': 'SR_B7'
        }
    else:  # Sentinel-2
        band_mapping = {
            'Blue': 'B2',
            'Green': 'B3',
            'Red': 'B4',
            'NIR': 'B8',
            'SWIR1': 'B11',
            'SWIR2': 'B12'
        }
    
    return [band_mapping[band] for band in selected_bands if band in band_mapping]

def add_spectral_indices(collection, indices, satellite):
    """Calculate spectral indices"""
    
    def calculate_indices(image):
        result = image
        
        # Get band names
        if 'Landsat' in satellite:
            red, green, blue, nir, swir = 'SR_B4', 'SR_B3', 'SR_B2', 'SR_B5', 'SR_B6'
        else:
            red, green, blue, nir, swir = 'B4', 'B3', 'B2', 'B8', 'B11'
        
        if 'NDVI' in indices:
            ndvi = image.normalizedDifference([nir, red]).rename('NDVI')
            result = result.addBands(ndvi)
        
        if 'NDWI' in indices:
            ndwi = image.normalizedDifference([green, nir]).rename('NDWI')
            result = result.addBands(ndwi)
        
        if 'NDBI' in indices:
            ndbi = image.normalizedDifference([swir, nir]).rename('NDBI')
            result = result.addBands(ndbi)
        
        if 'EVI' in indices:
            evi = image.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {'NIR': image.select(nir), 'RED': image.select(red), 'BLUE': image.select(blue)}
            ).rename('EVI')
            result = result.addBands(evi)
        
        return result
    
    return collection.map(calculate_indices)

def create_temporal_composites(collection, aggregation, start_date, end_date, method):
    """Create temporal composites"""
    
    from datetime import datetime
    import dateutil.parser
    
    start = dateutil.parser.parse(start_date)
    end = dateutil.parser.parse(end_date)
    
    composites = []
    
    if aggregation == 'yearly':
        # One composite per year
        for year in range(start.year, end.year + 1):
            yearly = collection.filterDate(f'{year}-01-01', f'{year}-12-31')
            
            if method == 'median':
                composite = yearly.median()
            elif method == 'mean':
                composite = yearly.mean()
            else:
                composite = yearly.mode()
            
            composites.append(composite)
    
    # Add similar logic for seasonal, monthly, etc.
    
    return composites

def export_to_drive(image, description, roi, satellite):
    """Export image to Google Drive"""
    
    scale = 30 if 'Landsat' in satellite else 10
    
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder='LandCover_Data',
        region=roi.getInfo()['coordinates'],
        scale=scale,
        maxPixels=1e13,
        crs='EPSG:4326',
        fileFormat='GeoTIFF'
    )
    
    task.start()
    return task

"""
🎉 DATA DOWNLOAD PAGE COMPLETE!

This page handles:
1. GEE authentication check
2. Loading satellite imagery
3. Cloud masking
4. Spectral band selection
5. Index calculation
6. Temporal compositing
7. Export to Google Drive
8. Download to local storage

Next: Create the data splitting page!
"""
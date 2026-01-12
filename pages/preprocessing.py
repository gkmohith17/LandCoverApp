"""
PART 3D: PREPROCESSING CONFIGURATION PAGE
==========================================

Save this as: pages/preprocessing.py

This page configures preprocessing options like band selection and spectral indices.
"""

import streamlit as st

def show():
    """Main function to show preprocessing configuration"""
    
    st.title("⚙️ Configure Preprocessing Options")
    
    st.markdown("""
    Configure how your satellite data will be processed before download.
    This includes selecting spectral bands, calculating indices, and normalization!
    """)
    
    # Get satellite from previous step
    satellite = st.session_state.timeline_config.get('satellite', 'Landsat 8')
    
    # ========================================================================
    # SPECTRAL BANDS SELECTION
    # ========================================================================
    
    st.subheader("🌈 Step 1: Select Spectral Bands")
    
    st.info("""
    **What are spectral bands?**
    
    Satellite sensors capture light in different wavelengths (colors). Each band 
    represents a different part of the light spectrum:
    - **Visible bands** (Blue, Green, Red) - What our eyes see
    - **Near-Infrared (NIR)** - Vegetation reflects strongly here
    - **Shortwave Infrared (SWIR)** - Sensitive to moisture and rocks
    """)
    
    # Define available bands based on satellite
    if "Landsat" in satellite:
        available_bands = {
            "Blue": "B2 - Good for water, atmosphere",
            "Green": "B3 - Good for vegetation",
            "Red": "B4 - Good for bare soil, urban",
            "NIR": "B5 - Near Infrared, vegetation health",
            "SWIR1": "B6 - Shortwave IR, moisture",
            "SWIR2": "B7 - Shortwave IR, minerals"
        }
    else:  # Sentinel-2
        available_bands = {
            "Blue": "B2 - Water, atmosphere",
            "Green": "B3 - Vegetation",
            "Red": "B4 - Bare soil, urban",
            "Red Edge 1": "B5 - Vegetation edge",
            "Red Edge 2": "B6 - Vegetation edge",
            "Red Edge 3": "B7 - Vegetation edge",
            "NIR": "B8 - Near Infrared",
            "SWIR1": "B11 - Moisture",
            "SWIR2": "B12 - Minerals"
        }
    
    # Show band selection
    st.markdown("**Select bands to include:**")
    
    # Recommend essential bands
    recommended = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
    
    selected_bands = []
    
    for band_name, description in available_bands.items():
        is_recommended = band_name in recommended
        default_value = is_recommended
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            is_selected = st.checkbox(
                band_name,
                value=default_value,
                key=f"band_{band_name}"
            )
            if is_selected:
                selected_bands.append(band_name)
        
        with col2:
            badge = "⭐ Recommended" if is_recommended else "Optional"
            st.caption(f"{description} - {badge}")
    
    if len(selected_bands) < 3:
        st.warning("⚠️ You should select at least 3 bands for meaningful classification!")
    else:
        st.success(f"✅ {len(selected_bands)} bands selected")
    
    # ========================================================================
    # SPECTRAL INDICES
    # ========================================================================
    
    st.markdown("---")
    st.subheader("📊 Step 2: Calculate Spectral Indices")
    
    st.info("""
    **What are spectral indices?**
    
    These are mathematical combinations of bands that highlight specific features:
    - **NDVI** - Vegetation health and density
    - **NDWI** - Water content and bodies
    - **NDBI** - Built-up (urban) areas
    - **EVI** - Enhanced vegetation (better in dense forests)
    
    Think of them as "super-powers" that make it easier to spot different land types!
    """)
    
    # Define available indices
    indices_info = {
        "NDVI": {
            "name": "Normalized Difference Vegetation Index",
            "formula": "(NIR - Red) / (NIR + Red)",
            "range": "-1 to +1",
            "interpretation": "> 0.3 = vegetation, < 0 = water/clouds",
            "requires": ["NIR", "Red"],
            "recommended": True
        },
        "NDWI": {
            "name": "Normalized Difference Water Index",
            "formula": "(Green - NIR) / (Green + NIR)",
            "range": "-1 to +1",
            "interpretation": "> 0 = water bodies, < 0 = land",
            "requires": ["Green", "NIR"],
            "recommended": True
        },
        "NDBI": {
            "name": "Normalized Difference Built-up Index",
            "formula": "(SWIR - NIR) / (SWIR + NIR)",
            "range": "-1 to +1",
            "interpretation": "> 0 = built-up areas, < 0 = vegetation",
            "requires": ["SWIR1", "NIR"],
            "recommended": True
        },
        "EVI": {
            "name": "Enhanced Vegetation Index",
            "formula": "2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))",
            "range": "-1 to +1",
            "interpretation": "Better than NDVI in dense vegetation",
            "requires": ["NIR", "Red", "Blue"],
            "recommended": True
        },
        "SAVI": {
            "name": "Soil Adjusted Vegetation Index",
            "formula": "((NIR - Red) / (NIR + Red + 0.5)) * 1.5",
            "range": "-1 to +1",
            "interpretation": "Better for areas with exposed soil",
            "requires": ["NIR", "Red"],
            "recommended": False
        },
        "MNDWI": {
            "name": "Modified Normalized Difference Water Index",
            "formula": "(Green - SWIR) / (Green + SWIR)",
            "range": "-1 to +1",
            "interpretation": "Better for urban water bodies",
            "requires": ["Green", "SWIR1"],
            "recommended": False
        }
    }
    
    selected_indices = []
    
    for index_key, index_info in indices_info.items():
        # Check if required bands are available
        can_calculate = all(band in selected_bands for band in index_info['requires'])
        
        with st.expander(f"{'⭐' if index_info['recommended'] else '📌'} {index_key} - {index_info['name']}"):
            
            st.markdown(f"""
            **Formula:** `{index_info['formula']}`
            
            **Range:** {index_info['range']}
            
            **What it shows:** {index_info['interpretation']}
            
            **Requires bands:** {', '.join(index_info['requires'])}
            """)
            
            if can_calculate:
                if st.checkbox(f"Calculate {index_key}", value=index_info['recommended'], key=f"index_{index_key}"):
                    selected_indices.append(index_key)
            else:
                st.warning(f"⚠️ Cannot calculate - missing required bands: {', '.join([b for b in index_info['requires'] if b not in selected_bands])}")
    
    if selected_indices:
        st.success(f"✅ {len(selected_indices)} indices will be calculated: {', '.join(selected_indices)}")
    else:
        st.info("ℹ️ No indices selected - only raw bands will be used")
    
    # ========================================================================
    # NORMALIZATION
    # ========================================================================
    
    st.markdown("---")
    st.subheader("📏 Step 3: Data Normalization")
    
    st.info("""
    **Why normalize?**
    
    Different bands have different value ranges. Normalization brings them to a common scale,
    which helps machine learning models train better!
    """)
    
    normalization = st.radio(
        "Choose normalization method:",
        [
            "Min-Max Scaling (0 to 1) - Recommended for Deep Learning",
            "Z-Score Standardization (mean=0, std=1) - Good for traditional ML",
            "No Normalization - Keep original values"
        ]
    )
    
    norm_method = normalization.split(" ")[0].lower()
    
    if "Min-Max" in normalization:
        st.markdown("""
        **Min-Max Scaling:**
        - Formula: `(value - min) / (max - min)`
        - Result: All values between 0 and 1
        - Best for: Neural networks, deep learning
        """)
    elif "Z-Score" in normalization:
        st.markdown("""
        **Z-Score Standardization:**
        - Formula: `(value - mean) / standard_deviation`
        - Result: Most values between -3 and +3
        - Best for: Traditional ML (SVM, Random Forest)
        """)
    else:
        st.warning("⚠️ Without normalization, models may train poorly due to different scales!")
    
    # ========================================================================
    # ADDITIONAL OPTIONS
    # ========================================================================
    
    st.markdown("---")
    st.subheader("🔧 Step 4: Additional Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        remove_na = st.checkbox(
            "Remove NA/NaN values",
            value=True,
            help="Remove pixels with missing data"
        )
        
        remove_outliers = st.checkbox(
            "Remove statistical outliers",
            value=False,
            help="Remove extreme values using IQR method"
        )
    
    with col2:
        apply_cloud_mask = st.checkbox(
            "Apply cloud masking",
            value=True,
            help="Mask out cloudy pixels (in addition to cloud cover filter)"
        )
        
        fill_gaps = st.checkbox(
            "Fill data gaps",
            value=False,
            help="Interpolate missing values"
        )
    
    # ========================================================================
    # SUMMARY & SAVE
    # ========================================================================
    
    st.markdown("---")
    st.subheader("✅ Configuration Summary")
    
    with st.container(border=True):
        st.markdown(f"""
        **Your preprocessing configuration:**
        
        - 🌈 **Bands:** {len(selected_bands)} selected ({', '.join(selected_bands)})
        - 📊 **Indices:** {len(selected_indices)} selected ({', '.join(selected_indices) if selected_indices else 'None'})
        - 📏 **Normalization:** {norm_method.title()}
        - 🔧 **Options:**
          - Remove NA: {'Yes' if remove_na else 'No'}
          - Remove outliers: {'Yes' if remove_outliers else 'No'}
          - Cloud masking: {'Yes' if apply_cloud_mask else 'No'}
          - Fill gaps: {'Yes' if fill_gaps else 'No'}
        
        **Total features per pixel:** {len(selected_bands) + len(selected_indices)}
        """)
    
    # Validate configuration
    if len(selected_bands) < 3:
        st.error("❌ Please select at least 3 spectral bands!")
        return
    
    # Navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Timeline"):
            st.session_state.current_step = 'timeline_config'
            st.rerun()
    
    with col2:
        if st.button("Start Download! 🚀", type="primary"):
            # Save configuration
            st.session_state.preprocessing_config = {
                'bands': selected_bands,
                'indices': selected_indices,
                'normalization': norm_method,
                'remove_na': remove_na,
                'remove_outliers': remove_outliers,
                'apply_cloud_mask': apply_cloud_mask,
                'fill_gaps': fill_gaps,
                'total_features': len(selected_bands) + len(selected_indices)
            }
            
            st.session_state.current_step = 'downloading'
            st.rerun()

"""
🎉 PREPROCESSING CONFIGURATION PAGE COMPLETE!

This page handles:
1. Spectral band selection
2. Spectral indices calculation
3. Normalization method
4. Additional preprocessing options

Next: Create the data download page (the exciting part!)
"""
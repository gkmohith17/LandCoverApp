"""
PART 3C: TIMELINE CONFIGURATION PAGE
=====================================

Save this as: pages/timeline_config.py

This page lets users configure the timeline and satellite selection.
"""

import streamlit as st
from datetime import datetime, date

def show():
    """Main function to show timeline configuration page"""
    
    st.title("📅 Configure Timeline & Satellite")
    
    st.markdown("""
    Choose which satellite to use and what time period to download data for.
    This determines the quality, resolution, and availability of your imagery!
    """)
    
    # ========================================================================
    # SATELLITE SELECTION
    # ========================================================================
    
    st.subheader("🛰️ Step 1: Choose Satellite")
    
    # Satellite comparison table
    with st.expander("📊 Compare Satellites (click to expand)", expanded=True):
        st.markdown("""
        | Feature | Landsat 7 | Landsat 8 | Sentinel-2 |
        |---------|-----------|-----------|------------|
        | **Launch Year** | 1999 | 2013 | 2015 |
        | **Resolution** | 30m | 30m | 10m/20m |
        | **Revisit Time** | 16 days | 16 days | 5 days |
        | **Spectral Bands** | 8 bands | 11 bands | 13 bands |
        | **Best For** | Long history | General use | High detail |
        | **Issues** | Scan line errors | None | Some gaps |
        | **Status** | ✅ Active | ✅ Active | ✅ Active |
        """)
    
    # Satellite selection
    satellite_options = {
        "Landsat 8 (Recommended for most uses)": {
            "name": "Landsat 8",
            "resolution": 30,
            "revisit": 16,
            "start_year": 2013,
            "description": "Best overall quality, no scan line errors, good spectral coverage"
        },
        "Sentinel-2 (Best resolution)": {
            "name": "Sentinel-2",
            "resolution": 10,
            "revisit": 5,
            "start_year": 2015,
            "description": "Highest resolution (10m), frequent revisits, great for urban areas"
        },
        "Landsat 7 (Longest history)": {
            "name": "Landsat 7",
            "resolution": 30,
            "revisit": 16,
            "start_year": 1999,
            "description": "Longest data record, has scan line errors after 2003"
        }
    }
    
    selected_satellite_key = st.radio(
        "Select Satellite:",
        list(satellite_options.keys()),
        help="Hover over options for more details"
    )
    
    satellite_info = satellite_options[selected_satellite_key]
    
    # Show detailed info about selected satellite
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Resolution", f"{satellite_info['resolution']}m")
    with col2:
        st.metric("Revisit Time", f"{satellite_info['revisit']} days")
    with col3:
        st.metric("Available Since", satellite_info['start_year'])
    with col4:
        years_available = 2024 - satellite_info['start_year']
        st.metric("Data Years", f"{years_available}+")
    
    st.info(f"💡 **Why choose {satellite_info['name']}?** {satellite_info['description']}")
    
    # ========================================================================
    # TIMELINE SELECTION
    # ========================================================================
    
    st.markdown("---")
    st.subheader("📅 Step 2: Choose Time Period")
    
    current_year = datetime.now().year
    min_year = satellite_info['start_year']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Start Date**")
        start_year = st.selectbox(
            "Year:",
            range(min_year, current_year + 1),
            index=len(range(min_year, current_year + 1)) - 7,  # Default to 6 years ago
            key="start_year"
        )
        start_month = st.selectbox(
            "Month:",
            range(1, 13),
            format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
            index=0,  # January
            key="start_month"
        )
    
    with col2:
        st.markdown("**End Date**")
        end_year = st.selectbox(
            "Year:",
            range(start_year, current_year + 1),
            index=len(range(start_year, current_year + 1)) - 1,  # Default to current year
            key="end_year"
        )
        end_month = st.selectbox(
            "Month:",
            range(1, 13),
            format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
            index=11,  # December
            key="end_month"
        )
    
    # Validate dates
    start_date = date(start_year, start_month, 1)
    end_date = date(end_year, end_month, 28)  # Simplified to avoid day issues
    
    if start_date >= end_date:
        st.error("❌ Start date must be before end date!")
        return
    
    # Calculate duration
    duration_days = (end_date - start_date).days
    duration_years = duration_days / 365.25
    
    st.success(f"✅ Time period: {duration_years:.1f} years ({duration_days} days)")
    
    # ========================================================================
    # TEMPORAL AGGREGATION
    # ========================================================================
    
    st.markdown("---")
    st.subheader("⏱️ Step 3: Temporal Aggregation")
    
    st.info("""
    **What is temporal aggregation?**
    
    Instead of downloading every single satellite image (which would be HUGE), 
    we create "composite" images by combining multiple images over a time period.
    
    For example: "Yearly" means we combine all images from each year into one image.
    """)
    
    aggregation = st.radio(
        "How should we combine images over time?",
        [
            "📅 Yearly (1 composite per year)",
            "🌸 Seasonal (4 composites per year)",
            "📆 Monthly (12 composites per year)",
            "⚙️ Custom Interval"
        ]
    )
    
    # Extract just the aggregation type
    agg_type = aggregation.split(" ")[1].lower().replace("(", "").replace(")", "")
    
    # Calculate number of composites
    if "Yearly" in aggregation:
        num_composites = int(duration_years) + 1
        st.info(f"📊 This will create **{num_composites} composite images** (one per year)")
    
    elif "Seasonal" in aggregation:
        num_composites = int(duration_years * 4)
        st.info(f"📊 This will create **~{num_composites} composite images** (4 per year: winter, spring, summer, fall)")
    
    elif "Monthly" in aggregation:
        num_composites = int(duration_years * 12)
        st.info(f"📊 This will create **~{num_composites} composite images** (12 per year)")
    
    elif "Custom" in aggregation:
        interval_days = st.slider(
            "Interval (days):",
            min_value=1,
            max_value=365,
            value=90,
            help="Create one composite every X days"
        )
        num_composites = duration_days // interval_days
        st.info(f"📊 This will create **~{num_composites} composite images** (one every {interval_days} days)")
    
    # Compositing method
    st.markdown("**Compositing Method:**")
    composite_method = st.radio(
        "How should we combine multiple images into one?",
        [
            "Median (Recommended - reduces noise and clouds)",
            "Mean (Average all pixel values)",
            "Mode (Most common value)"
        ],
        help="""
        - **Median**: Best for removing clouds and outliers
        - **Mean**: Smoother but affected by clouds
        - **Mode**: Good for discrete values
        """
    )
    
    # ========================================================================
    # ESTIMATE DATA SIZE
    # ========================================================================
    
    st.markdown("---")
    st.subheader("💾 Estimated Data Size")
    
    # Rough estimation
    # Landsat ~100MB per scene, Sentinel ~500MB per scene
    # After processing and compositing, much smaller
    
    base_size_per_composite = 50 if "Landsat" in satellite_info['name'] else 100  # MB
    
    # Adjust by region size
    if st.session_state.region_area_km2:
        area = st.session_state.region_area_km2
        size_multiplier = min(area / 10000, 2)  # Max 2x for large areas
    else:
        size_multiplier = 1
    
    estimated_size_mb = num_composites * base_size_per_composite * size_multiplier
    estimated_size_gb = estimated_size_mb / 1024
    
    # Estimated download time (assuming 10 Mbps internet)
    download_time_minutes = (estimated_size_mb * 8) / (10 * 60)  # Convert to minutes
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Composites",
            f"{num_composites}",
            help="Number of composite images to download"
        )
    
    with col2:
        if estimated_size_gb < 1:
            st.metric(
                "Estimated Size",
                f"{estimated_size_mb:.0f} MB"
            )
        else:
            st.metric(
                "Estimated Size",
                f"{estimated_size_gb:.2f} GB"
            )
    
    with col3:
        if download_time_minutes < 60:
            st.metric(
                "Download Time",
                f"~{download_time_minutes:.0f} min",
                help="Estimated time (depends on internet speed)"
            )
        else:
            st.metric(
                "Download Time",
                f"~{download_time_minutes/60:.1f} hours"
            )
    
    # Warning for large downloads
    if estimated_size_gb > 5:
        st.warning("""
        ⚠️ **Large Download Warning**
        
        This configuration will download a lot of data (> 5GB). Consider:
        - Reducing the time period
        - Using yearly instead of monthly aggregation
        - Selecting a smaller region
        - Ensuring you have enough disk space
        """)
    
    # ========================================================================
    # CLOUD COVER FILTER
    # ========================================================================
    
    st.markdown("---")
    st.subheader("☁️ Cloud Filter (Optional)")
    
    enable_cloud_filter = st.checkbox(
        "Filter out cloudy images",
        value=True,
        help="Only include images with cloud cover below threshold"
    )
    
    if enable_cloud_filter:
        cloud_threshold = st.slider(
            "Maximum cloud cover (%)",
            min_value=0,
            max_value=100,
            value=20,
            help="Images with more than this % clouds will be filtered out"
        )
        st.info(f"ℹ️ Only images with less than {cloud_threshold}% cloud cover will be used")
    else:
        cloud_threshold = 100
        st.warning("⚠️ Warning: Without cloud filtering, your images may have clouds")
    
    # ========================================================================
    # SAVE CONFIGURATION
    # ========================================================================
    
    st.markdown("---")
    st.subheader("✅ Review Configuration")
    
    # Display summary
    with st.container(border=True):
        st.markdown(f"""
        **Summary of your configuration:**
        
        - 🛰️ **Satellite:** {satellite_info['name']}
        - 📅 **Time Period:** {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}
        - ⏱️ **Duration:** {duration_years:.1f} years
        - 📊 **Aggregation:** {agg_type.title()}
        - 🖼️ **Composites:** {num_composites} images
        - ☁️ **Cloud Threshold:** {cloud_threshold}%
        - 💾 **Estimated Size:** {estimated_size_gb:.2f} GB
        """)
    
    # Save button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Region Selection"):
            st.session_state.current_step = 'region_selection'
            st.rerun()
    
    with col2:
        if st.button("Next: Preprocessing Options →", type="primary"):
            # Save configuration
            st.session_state.timeline_config = {
                'satellite': satellite_info['name'],
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'aggregation': agg_type,
                'num_composites': num_composites,
                'composite_method': composite_method.split(" ")[0].lower(),
                'cloud_threshold': cloud_threshold,
                'estimated_size_gb': estimated_size_gb
            }
            
            st.session_state.current_step = 'preprocessing'
            st.rerun()

"""
🎉 TIMELINE CONFIGURATION PAGE COMPLETE!

This page handles:
1. Satellite selection (Landsat 7/8, Sentinel-2)
2. Time period selection
3. Temporal aggregation settings
4. Data size estimation
5. Cloud filtering options

Next: Create the preprocessing configuration page!
"""
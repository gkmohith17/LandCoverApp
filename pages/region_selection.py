"""
PART 3B: REGION SELECTION PAGE
===============================

Save this as: pages/region_selection.py

This page lets users select their region of interest for downloading satellite data.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from pathlib import Path
import json
import geopandas as gpd
from shapely.geometry import shape, mapping

def show():
    """Main function to show region selection page"""
    
    st.title("🗺️ Select Your Region of Interest")
    
    st.markdown("""
    Choose the geographic area where you want to download satellite imagery.
    You have multiple options depending on your needs!
    """)
    
    # ========================================================================
    # Selection Method Choice
    # ========================================================================
    
    st.subheader("How would you like to define your region?")
    
    selection_method = st.radio(
        "Choose a method:",
        [
            "🖱️ Draw on Interactive Map (Easiest)",
            "📍 Enter Coordinates Manually",
            "📁 Upload Shapefile",
            "🇮🇳 Select from Indian States/Districts"
        ],
        help="""
        - **Draw on Map**: Click and drag to draw your region
        - **Coordinates**: Enter lat/lon coordinates
        - **Shapefile**: Upload .shp file with your region
        - **States/Districts**: Choose from pre-defined Indian boundaries
        """
    )
    
    st.markdown("---")
    
    # ========================================================================
    # Method 1: Interactive Map Drawing
    # ========================================================================
    
    if "Draw on Interactive Map" in selection_method:
        show_interactive_map_selection()
    
    # ========================================================================
    # Method 2: Manual Coordinates
    # ========================================================================
    
    elif "Enter Coordinates" in selection_method:
        show_coordinate_input()
    
    # ========================================================================
    # Method 3: Upload Shapefile
    # ========================================================================
    
    elif "Upload Shapefile" in selection_method:
        show_shapefile_upload()
    
    # ========================================================================
    # Method 4: Indian States/Districts
    # ========================================================================
    
    elif "Indian States" in selection_method:
        show_indian_boundaries()
    
    # ========================================================================
    # Navigation
    # ========================================================================
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back"):
            st.session_state.current_step = 'dataset_choice'
            st.rerun()
    
    with col2:
        if st.session_state.region_geometry is not None:
            if st.button("Next: Timeline Configuration →", type="primary"):
                st.session_state.current_step = 'timeline_config'
                st.rerun()

# ============================================================================
# METHOD 1: INTERACTIVE MAP DRAWING
# ============================================================================

def show_interactive_map_selection():
    """Show interactive map for drawing region"""
    
    st.subheader("🖱️ Draw Your Region on the Map")
    
    st.info("""
    **How to use:**
    1. Click the ⬛ (rectangle) or ⬟ (polygon) tool in the top-left
    2. Draw your region on the map
    3. Click 'Save Drawing' when done
    4. You can redraw by clicking the 🗑️ (trash) icon
    """)
    
    # Create base map centered on India
    m = folium.Map(
        location=[20.5937, 78.9629],  # Center of India
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Add drawing tools
    from folium.plugins import Draw
    
    Draw(
        export=False,
        position='topleft',
        draw_options={
            'polyline': False,      # Don't allow lines
            'circle': False,         # Don't allow circles
            'circlemarker': False,   # Don't allow circle markers
            'marker': False,         # Don't allow point markers
            'polygon': True,         # Allow polygons ⬟
            'rectangle': True        # Allow rectangles ⬛
        },
        edit_options={
            'edit': True,    # Allow editing drawn shapes
            'remove': True   # Allow deleting shapes
        }
    ).add_to(m)
    
    # Add a scale bar
    from folium.plugins import MeasureControl
    MeasureControl(position='bottomleft').add_to(m)
    
    # Display map
    map_data = st_folium(
        m,
        width=700,
        height=500,
        returned_objects=["all_drawings"]
    )
    
    # Check if user drew something
    if map_data and map_data.get('all_drawings'):
        if len(map_data['all_drawings']) > 0:
            
            # Get the last drawn shape
            drawn_shape = map_data['all_drawings'][-1]
            geometry = drawn_shape['geometry']
            
            # Convert to proper format
            region_geometry = {
                'type': geometry['type'],
                'coordinates': geometry['coordinates']
            }
            
            # Calculate area
            from shapely.geometry import shape
            shape_obj = shape(geometry)
            area_km2 = shape_obj.area * 111 * 111  # Rough conversion to km²
            
            # Show region info
            st.success("✅ Region drawn successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Area", f"{area_km2:.2f} km²")
            
            with col2:
                bounds = shape_obj.bounds  # (minx, miny, maxx, maxy)
                st.metric("Bounds", f"{len(bounds)} coords")
            
            with col3:
                if area_km2 > 50000:
                    st.warning("⚠️ Large area!")
                else:
                    st.info("✅ Good size")
            
            # Warning for very large areas
            if area_km2 > 50000:
                st.warning("""
                ⚠️ **Large Area Warning**
                
                Your selected region is very large (>50,000 km²). This might:
                - Take a long time to download
                - Use a lot of storage space
                - Hit Google Earth Engine memory limits
                
                **Recommendation:** Consider selecting a smaller region or splitting into tiles.
                """)
            
            # Save button
            if st.button("💾 Save This Region", type="primary"):
                st.session_state.region_geometry = region_geometry
                st.session_state.region_area_km2 = area_km2
                st.success("✅ Region saved! Click 'Next' below to continue.")

# ============================================================================
# METHOD 2: MANUAL COORDINATES
# ============================================================================

def show_coordinate_input():
    """Show form for manual coordinate input"""
    
    st.subheader("📍 Enter Coordinates Manually")
    
    st.info("""
    Enter the bounding box coordinates for your region.
    Format: Latitude and Longitude in decimal degrees.
    
    **Example for Bangalore, India:**
    - Min Latitude: 12.8
    - Max Latitude: 13.2
    - Min Longitude: 77.4
    - Max Longitude: 77.8
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Latitude (North-South)**")
        min_lat = st.number_input(
            "Minimum Latitude (South)",
            min_value=-90.0,
            max_value=90.0,
            value=12.8,
            step=0.1,
            format="%.6f"
        )
        max_lat = st.number_input(
            "Maximum Latitude (North)",
            min_value=-90.0,
            max_value=90.0,
            value=13.2,
            step=0.1,
            format="%.6f"
        )
    
    with col2:
        st.markdown("**Longitude (East-West)**")
        min_lon = st.number_input(
            "Minimum Longitude (West)",
            min_value=-180.0,
            max_value=180.0,
            value=77.4,
            step=0.1,
            format="%.6f"
        )
        max_lon = st.number_input(
            "Maximum Longitude (East)",
            min_value=-180.0,
            max_value=180.0,
            value=77.8,
            step=0.1,
            format="%.6f"
        )
    
    # Validate coordinates
    if min_lat >= max_lat:
        st.error("❌ Minimum latitude must be less than maximum latitude!")
        return
    
    if min_lon >= max_lon:
        st.error("❌ Minimum longitude must be less than maximum longitude!")
        return
    
    # Create geometry
    region_geometry = {
        'type': 'Polygon',
        'coordinates': [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat]
        ]]
    }
    
    # Calculate area
    from shapely.geometry import shape
    shape_obj = shape(region_geometry)
    area_km2 = shape_obj.area * 111 * 111
    
    # Show preview map
    st.subheader("📍 Preview Your Region")
    
    preview_map = folium.Map(
        location=[(min_lat + max_lat)/2, (min_lon + max_lon)/2],
        zoom_start=10
    )
    
    # Add rectangle
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        color='red',
        fill=True,
        fillOpacity=0.2,
        popup=f"Area: {area_km2:.2f} km²"
    ).add_to(preview_map)
    
    st_folium(preview_map, width=700, height=400)
    
    # Show info
    st.metric("Estimated Area", f"{area_km2:.2f} km²")
    
    # Save button
    if st.button("💾 Use These Coordinates", type="primary"):
        st.session_state.region_geometry = region_geometry
        st.session_state.region_area_km2 = area_km2
        st.success("✅ Coordinates saved! Click 'Next' below to continue.")

# ============================================================================
# METHOD 3: SHAPEFILE UPLOAD
# ============================================================================

def show_shapefile_upload():
    """Show shapefile upload interface"""
    
    st.subheader("📁 Upload Shapefile")
    
    st.info("""
    **What is a Shapefile?**
    
    A shapefile is a common geographic data format. It actually consists of 
    multiple files:
    - `.shp` - The main file with geometry
    - `.shx` - Index file
    - `.dbf` - Attribute data
    - `.prj` - Projection info (optional)
    
    You need to upload at least the first 3 files!
    
    **Where to get shapefiles:**
    - GADM: https://gadm.org/ (Free administrative boundaries)
    - OpenStreetMap: https://www.openstreetmap.org/
    - Your own GIS software (QGIS, ArcGIS)
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload shapefile components (.shp, .shx, .dbf, .prj)",
        type=['shp', 'shx', 'dbf', 'prj'],
        accept_multiple_files=True,
        help="Upload all files that came with your shapefile"
    )
    
    if uploaded_files and len(uploaded_files) >= 3:
        
        # Create temp directory
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        
        # Save all uploaded files
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
        
        # Find the .shp file
        shp_file = next((f for f in temp_dir.glob('*.shp')), None)
        
        if shp_file:
            try:
                # Read shapefile
                gdf = gpd.read_file(shp_file)
                
                st.success(f"✅ Loaded {len(gdf)} feature(s) from shapefile!")
                
                # Show attributes
                st.subheader("📋 Shapefile Attributes")
                st.dataframe(gdf.head(), use_container_width=True)
                
                # Let user select which feature to use if multiple
                if len(gdf) > 1:
                    st.warning(f"⚠️ Your shapefile contains {len(gdf)} features.")
                    feature_idx = st.selectbox(
                        "Select which feature to use:",
                        range(len(gdf)),
                        format_func=lambda x: f"Feature {x+1}"
                    )
                else:
                    feature_idx = 0
                
                # Get geometry
                selected_geom = gdf.iloc[feature_idx].geometry
                
                # Convert to proper CRS (WGS84)
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf_wgs = gdf.to_crs(epsg=4326)
                    selected_geom = gdf_wgs.iloc[feature_idx].geometry
                
                # Show on map
                st.subheader("🗺️ Preview")
                
                bounds = selected_geom.bounds
                center_lat = (bounds[1] + bounds[3]) / 2
                center_lon = (bounds[0] + bounds[2]) / 2
                
                preview_map = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=10
                )
                
                # Add geometry to map
                folium.GeoJson(
                    mapping(selected_geom),
                    style_function=lambda x: {
                        'fillColor': 'blue',
                        'color': 'blue',
                        'weight': 2,
                        'fillOpacity': 0.3
                    }
                ).add_to(preview_map)
                
                st_folium(preview_map, width=700, height=400)
                
                # Calculate area
                area_km2 = selected_geom.area * 111 * 111
                st.metric("Area", f"{area_km2:.2f} km²")
                
                # Save button
                if st.button("💾 Use This Shapefile", type="primary"):
                    region_geometry = mapping(selected_geom)
                    st.session_state.region_geometry = region_geometry
                    st.session_state.region_area_km2 = area_km2
                    st.success("✅ Shapefile saved! Click 'Next' below to continue.")
                
            except Exception as e:
                st.error(f"❌ Error reading shapefile: {str(e)}")
                st.exception(e)
        else:
            st.error("❌ No .shp file found in uploaded files!")

# ============================================================================
# METHOD 4: INDIAN STATES/DISTRICTS
# ============================================================================

def show_indian_boundaries():
    """Show dropdown for Indian states and districts"""
    
    st.subheader("🇮🇳 Select Indian State/District")
    
    st.info("""
    Choose from pre-defined administrative boundaries of India.
    This is the easiest option if you're working in India!
    """)
    
    # Simplified Indian states for demo
    # In production, load from actual boundary files
    indian_states = {
        "Karnataka": {"lat": 15.3173, "lon": 75.7139},
        "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569},
        "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
        "Kerala": {"lat": 10.8505, "lon": 76.2711},
        "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400},
        # Add more states...
    }
    
    state = st.selectbox(
        "Select State:",
        list(indian_states.keys())
    )
    
    st.info(f"""
    **Note:** For this demo, we're using simplified boundaries.
    
    **For production:**
    - Download Indian boundary shapefiles from:
      - GADM: https://gadm.org/download_country.html (Select India)
      - DataMeet: https://github.com/datameet/maps
    - Place them in `data/boundaries/` folder
    - The app will automatically load them
    """)
    
    # Create a simple bounding box for the selected state
    # In production, load actual boundary from shapefile
    state_info = indian_states[state]
    
    # Create approximate bounding box (±0.5 degrees)
    region_geometry = {
        'type': 'Polygon',
        'coordinates': [[
            [state_info['lon'] - 2, state_info['lat'] - 2],
            [state_info['lon'] + 2, state_info['lat'] - 2],
            [state_info['lon'] + 2, state_info['lat'] + 2],
            [state_info['lon'] - 2, state_info['lat'] + 2],
            [state_info['lon'] - 2, state_info['lat'] - 2]
        ]]
    }
    
    # Show preview
    preview_map = folium.Map(
        location=[state_info['lat'], state_info['lon']],
        zoom_start=7
    )
    
    folium.GeoJson(region_geometry).add_to(preview_map)
    
    st_folium(preview_map, width=700, height=400)
    
    # Save button
    if st.button(f"💾 Use {state} Boundary", type="primary"):
        from shapely.geometry import shape
        shape_obj = shape(region_geometry)
        area_km2 = shape_obj.area * 111 * 111
        
        st.session_state.region_geometry = region_geometry
        st.session_state.region_area_km2 = area_km2
        st.success(f"✅ {state} boundary saved! Click 'Next' below to continue.")

"""
🎉 REGION SELECTION PAGE COMPLETE!

This page provides 4 ways to select a region:
1. Draw on interactive map (easiest)
2. Enter coordinates manually
3. Upload shapefile
4. Select from Indian states

Next: Create the timeline configuration page!
"""
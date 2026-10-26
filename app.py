import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
from google.oauth2 import service_account
from geopy.geocoders import Nominatim

# Setup the webpage layout
st.set_page_config(page_title="Methane AI Predictor", layout="wide")
st.title("🌍 Methane AI Predictor (SAR Data)")
st.write("**Indradhanu PCCOE IGC 2026 Prototype**")

# 1. Securely log into Google Earth Engine using Streamlit Secrets
@st.cache_resource
def init_ee():
    key_dict = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(key_dict)
    ee.Initialize(credentials=creds, project='ai-methane-model')

try:
    init_ee()
except Exception as e:
    st.error("Failed to connect to Google Earth Engine. Please check your secrets.")
    st.stop()

# 2. Ask the user for a location
place_name = st.text_input("Enter a city or region name:", "Jalandhar, Punjab")

if place_name:
    geolocator = Nominatim(user_agent="methane_hackathon_app")
    location = geolocator.geocode(place_name)

    if location:
        st.success(f"Location found: {location.address}")
        lat, lon = location.latitude, location.longitude

        with st.spinner("Fetching Satellite Data & Running AI Model... (This takes about 20 seconds)"):
            
            # Create a 15km bounding box around the selected city
            point = ee.Geometry.Point([lon, lat])
            roi = point.buffer(15000).bounds()
            
            # Load S1 (SAR - Radar Textures)
            s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterBounds(roi)
                  .filterDate('2023-01-01', '2023-12-31')
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .select(['VV'])
                  .mean()
                  .clip(roi))
            
            # Load S5P (Methane - Answer Key)
            s5p = (ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4')
                   .filterBounds(roi)
                   .filterDate('2023-01-01', '2023-12-31')
                   .select('CH4_column_volume_mixing_ratio_dry_air')
                   .mean()
                   .clip(roi))

            # Train the AI Model dynamically on the selected region
            combined_image = s1.addBands(s5p)
            training_data = combined_image.sample(region=roi, scale=1000, numPixels=500, seed=42)
            rf_model = ee.Classifier.smileRandomForest(10).setOutputMode('REGRESSION').train(
                features=training_data, classProperty='CH4_column_volume_mixing_ratio_dry_air', inputProperties=['VV']
            )
            predicted_methane = s1.classify(rf_model)

        # 3. Create the Interactive Map
        map_view = folium.Map(location=[lat, lon], zoom_start=11, tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google')
        
        def add_ee_layer(self, ee_image_object, vis_params, name):
            map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Map Data &copy; Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
        folium.Map.add_ee_layer = add_ee_layer

        vis_params = {'min': 1800, 'max': 1950, 'palette': ['blue', 'cyan', 'green', 'yellow', 'red']}
        
        # Add layers
        map_view.add_ee_layer(predicted_methane, vis_params, 'AI Predicted Methane (SAR)')
        map_view.add_ee_layer(s5p, vis_params, 'Real Methane Ground Truth (S5P)')
        folium.LayerControl().add_to(map_view)

        # Display the map on the webpage
        st_folium(map_view, width=800, height=500)

        # 4. Tips Section
        st.subheader("💡 Methane Reduction Tips for this Region")
        st.write("1. **Waste Management:** Divert organic waste from local landfills into composting.")
        st.write("2. **Agriculture:** Implement alternate wetting and drying for local crops.")
        st.write("3. **Infrastructure:** Inspect local gas pipelines for micro-leaks using drone SAR.")
        
    else:
        st.error("❌ Location not found. Please try spelling it differently.")

import streamlit as st
import joblib
import pandas as pd
import datetime

# Define the CSS to hide the toolbar
hide_toolbar_css = """
<style>
[data-testid="stToolbar"] {
    display: none;
}
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(hide_toolbar_css, unsafe_allow_html=True)



# App Title
st.title("🌌 Space Traffic Density Prediction")

# Sidebar for Model Selection
st.sidebar.header("Model Selection")
try:
    model_options = {
        'Random Forest Regressor': 'model/RandomForestRegressor.joblib',
        'Linear Regression': 'model/LinearRegression.joblib',
        'Support Vector Regressor': 'model/SVR.joblib',
    }
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))
    model_path = model_options[selected_model]
    model = joblib.load(model_path)
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Main Section
st.header("📝 Input Details")

# Date Input
selected_date = st.date_input(
    "🗓️ Select a Date",
    min_value=datetime.date(2000, 1, 1),
    max_value=datetime.date(2100, 12, 31),
    value=datetime.date(2024, 12, 19),
    help="Choose the date for which you want to predict traffic density.",
)
year, month, day = selected_date.year, selected_date.month, selected_date.day

# Location Selection
try:
    location_encoder = joblib.load('model/label_encoder.joblib')
    locations = list(location_encoder.classes_)
    selected_location = st.selectbox("🌍 Select Location", locations)
except Exception as e:
    st.error(f"Error loading locations: {e}")
    st.stop()

# Object Types with Toggle Buttons in a 3x2 Layout
st.header("🛰️ Object Types")
col1, col2 = st.columns(2)

with col1:
    asteroid_mining_ship = st.toggle("Asteroid Mining Ship", value=False)
    manned_spacecraft = st.toggle("Manned Spacecraft", value=False)
    satellite = st.toggle("Satellite", value=False)

with col2:
    scientific_probe = st.toggle("Scientific Probe", value=False)
    space_debris = st.toggle("Space Debris", value=False)
    space_station = st.toggle("Space Station", value=True)

# Mapping toggles to object types
object_types = {
    "Object_Type_Asteroid Mining Ship": asteroid_mining_ship,
    "Object_Type_Manned Spacecraft": manned_spacecraft,
    "Object_Type_Satellite": satellite,
    "Object_Type_Scientific Probe": scientific_probe,
    "Object_Type_Space Debris": space_debris,
    "Object_Type_Space Station": space_station,
}

# Centered Prediction Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔮 Predict Traffic Density"):
        # Prepare input data
        input_data = {
            'Location': [selected_location],
            'Year': [year],
            'Month': [month],
            'Day': [day],
            **{key: int(value) for key, value in object_types.items()},
        }
        input_df = pd.DataFrame(input_data)

        # Encode Location
        try:
            input_df['Location_Encoded'] = location_encoder.transform(input_df['Location'])
            input_df = input_df.drop('Location', axis=1)
        except ValueError as e:
            st.error(f"Location encoding error: {e}")
            st.stop()

        # Reorder columns to match the training data
        columns_order = [
            'Location_Encoded', 'Year', 'Month', 'Day',
            'Object_Type_Asteroid Mining Ship', 'Object_Type_Manned Spacecraft',
            'Object_Type_Satellite', 'Object_Type_Scientific Probe',
            'Object_Type_Space Debris', 'Object_Type_Space Station'
        ]
        input_df = input_df[columns_order]

        # Make Prediction
        try:
            prediction = model.predict(input_df)
            st.success(f"🌟 Predicted Traffic Density: **{prediction[0]:.2f}**")
        except ValueError as e:
            st.error(f"Prediction Error: {e}")

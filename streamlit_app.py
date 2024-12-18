import streamlit as st
import joblib
import pandas as pd

# App title
st.title("Space Traffic Density Prediction")

# Sidebar for user inputs
st.sidebar.header("Input Features")

# Dropdown to select model
model_options = {
    'Random Forest Regressor': 'model/RandomForestRegressor.joblib',
    'Linear Regression': 'model/LinearRegression.joblib', 
    'Support Vector Regressor': 'model/SVR.joblib',
}
selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))

# Load the selected model
model_path = model_options[selected_model]
model = joblib.load(model_path)

# Load the trained LabelEncoder
location_encoder = joblib.load('model/label_encoder.joblib')

# Dropdown for location
locations = list(location_encoder.classes_)
selected_location = st.sidebar.selectbox("Location", locations)

# Input fields for Year, Month, and Day
year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2024)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=10)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=21)

# Checkboxes for Object Types
object_types = {
    "Object_Type_Asteroid Mining Ship": st.sidebar.checkbox("Asteroid Mining Ship", value=False),
    "Object_Type_Manned Spacecraft": st.sidebar.checkbox("Manned Spacecraft", value=False),
    "Object_Type_Satellite": st.sidebar.checkbox("Satellite", value=False),
    "Object_Type_Scientific Probe": st.sidebar.checkbox("Scientific Probe", value=False),
    "Object_Type_Space Debris": st.sidebar.checkbox("Space Debris", value=False),
    "Object_Type_Space Station": st.sidebar.checkbox("Space Station", value=True),
}

# Prediction button
if st.sidebar.button("Predict Traffic Density"):
    # Prepare input data for prediction
    input_data = {
        'Location': [selected_location],
        'Year': [year],
        'Month': [month],
        'Day': [day],
        **object_types
    }

    # Convert the input data into a pandas DataFrame
    input_df = pd.DataFrame(input_data)

    # Encode the 'Location' column using the loaded LabelEncoder
    try:
        input_df['Location_Encoded'] = location_encoder.transform(input_df['Location'])
    except ValueError as e:
        st.error(f"Error encoding location: {e}. Ensure the input data matches the training data locations.")
        st.stop()

    # Drop the 'Location' column (as it's no longer needed after encoding)
    input_df = input_df.drop('Location', axis=1)

    # Ensure the column order matches the model's training data
    columns_order = [
        'Location_Encoded', 'Year', 'Month', 'Day',
        'Object_Type_Asteroid Mining Ship', 'Object_Type_Manned Spacecraft',
        'Object_Type_Satellite', 'Object_Type_Scientific Probe',
        'Object_Type_Space Debris', 'Object_Type_Space Station'
    ]
    input_df = input_df[columns_order]  # Reorder columns to match training

    # Make prediction using the trained model
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Traffic Density: {prediction[0]:.2f}")
    except ValueError as e:
        st.error(f"Prediction Error: {e}. Ensure the input matches the training data schema.")

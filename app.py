import streamlit as st
import numpy as np
import pickle

def load_model():
    # Load the logistic regression model from the pickle file
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def dig_page():
    st.header("Dig Page")
    st.subheader("Lidar Surface Image")
    lidar_image = st.image("dragline.png", caption="Bucket current status", use_column_width=True)

    st.subheader("Digging Parameters")

    with st.form(key="digging_parameters_form"):
        # Set default values for Bucket Capacity, Maximum Digging Depth, and Boom Length
        bucket_capacity = st.number_input("Bucket Capacity (cubic meters)", min_value=16.0, step=0.0, value= 24.0)
        present_volume = st.number_input("Present Volume (cubic meters)", min_value=0.0, step=0.1, value=17.60)
        max_dig_depth = st.number_input("Maximum Digging Depth (meters)", 25.0)
        hoist_speed = st.number_input("Hoist Speed (meters per second)", min_value=0.0, step=0.1, value=112.89)
        swing_speed = st.number_input("Swing Speed (degrees per second)", min_value=0.0, step=0.1, value=89.22)
        boom_length = st.number_input("Boom Length (meters)", 96.0)

        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        # Load the machine learning model
        model = load_model()

        # Prepare the input features for prediction
        # Add placeholders for the missing features if needed
        features = np.array([present_volume, hoist_speed, swing_speed, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1)

        # Make a prediction using the loaded model
        prediction = model.predict(features)

        # Display the prediction result
        if prediction == 0:
            st.success("Start digging")
        else:
            st.success("Stop digging")
        # st.success(f"Prediction: {prediction}")

# Call the function to run the Streamlit app
dig_page()

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('Aircraft Damage Propagation (1).pkl')


# Main function to run the app
def main():
    st.title('Aircraft Prediction')
    # Load the pre-trained model
    model = load_model()

    # Prediction form
    st.subheader('Make a Prediction')
    unit_number = st.number_input('unit_number', min_value=0.0)
    time_cycles = st.number_input('time_cycles', min_value=0.0)
    setting_1 = st.number_input('setting_1', min_value=0.0)
    setting_2 = st.number_input('setting_2', min_value=0.0)
    setting_3 = st.number_input('setting_3', min_value=0.0)
    sensor_1 = st.number_input('s_1', min_value=0.0)
    sensor_2 = st.number_input('s_2', min_value=0.0)
    sensor_3 = st.number_input('s_3', min_value=0.0)
    sensor_4 = st.number_input('s_4', min_value=0.0)
    sensor_5 = st.number_input('s_5', min_value=0.0)
    sensor_6 = st.number_input('s_6', min_value=0.0)
    sensor_7 = st.number_input('s_7', min_value=0.0)
    sensor_8 = st.number_input('s_8', min_value=0.0)
    sensor_9 = st.number_input('s_9', min_value=0.0)
    sensor_10 = st.number_input('s_10', min_value=0.0)
    sensor_11 = st.number_input('s_11', min_value=0.0)
    sensor_12 = st.number_input('s_12', min_value=0.0)
    sensor_13 = st.number_input('s_13', min_value=0.0)
    sensor_14 = st.number_input('s_14', min_value=0.0)
    sensor_15 = st.number_input('s_15', min_value=0.0)
    sensor_16 = st.number_input('s_16', min_value=0.0)
    sensor_17 = st.number_input('s_17', min_value=0.0)
    sensor_18 = st.number_input('s_18', min_value=0.0)
    sensor_19 = st.number_input('s_19', min_value=0.0)
    sensor_20 = st.number_input('s_20', min_value=0.0)
    sensor_21 = st.number_input('s_21', min_value=0.0)

    if st.button('Predict'):
        prediction = model.predict([[unit_number, time_cycles, setting_1, setting_2, setting_3,
                                     sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9,
                                     sensor_10, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_16, sensor_17, sensor_18,
                                     sensor_19, sensor_20, sensor_21]])
        st.success(f'Aircraft prediction: {prediction[0]}')


if __name__ == '__main__':
    main()

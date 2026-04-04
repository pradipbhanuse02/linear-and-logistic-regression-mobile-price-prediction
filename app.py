import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model, scaler, and features
model = joblib.load('cellphone_price_model.pkl')
scaler = joblib.load('cellphone_scaler.pkl')
features = joblib.load('cellphone_features.pkl')

st.set_page_config(page_title="Phone Price Predictor", layout="wide")
st.title("📱 Cellphone Price Prediction Model")

st.markdown("""
This app uses a machine learning model trained on cellphone specifications to predict the price.
Enter the phone specifications below to get a price prediction.
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Phone Specifications")
    
    product_id = st.number_input("Product ID", min_value=0, max_value=1000, value=1)
    sale = st.number_input("Number of Sales", min_value=0, max_value=1000000, value=1000)
    weight = st.number_input("Weight (g)", min_value=50.0, max_value=300.0, value=150.0)
    resolution = st.number_input("Resolution (ppi)", min_value=100.0, max_value=600.0, value=300.0)
    ppi = st.number_input("Pixels Per Inch (PPI)", min_value=100.0, max_value=600.0, value=300.0)

with col2:
    cpu_core = st.number_input("CPU Cores", min_value=1, max_value=16, value=4)
    cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.5, max_value=3.5, value=2.0)
    internal_mem = st.number_input("Internal Memory (GB)", min_value=8, max_value=1024, value=64)
    ram = st.number_input("RAM (GB)", min_value=2, max_value=16, value=4)
    rear_cam = st.number_input("Rear Camera (MP)", min_value=2, max_value=300, value=12)

col3, col4 = st.columns(2)

with col3:
    front_cam = st.number_input("Front Camera (MP)", min_value=2, max_value=100, value=8)
    battery = st.number_input("Battery (mAh)", min_value=1000, max_value=10000, value=4000)
    thickness = st.number_input("Thickness (mm)", min_value=5.0, max_value=15.0, value=8.0)

# Prepare input for prediction
if st.button("🔮 Predict Price", key="predict_btn", use_container_width=True):
    try:
        # Create a dictionary with all features
        input_data = {
            'Product_id': product_id,
            'Sale': sale,
            'weight': weight,
            'resolution': resolution,
            'ppi': ppi,
            'cpu core': cpu_core,
            'cpu freq': cpu_freq,
            'internal mem': internal_mem,
            'ram': ram,
            'RearCam': rear_cam,
            'Front_Cam': front_cam,
            'battery': battery,
            'thickness': thickness
        }
        
        # Create input array in the same order as feature_cols
        input_array = []
        for feature in features:
            if '_log1p' in feature:
                # Handle log1p transformed features
                base_feature = feature.replace('_log1p', '')
                if base_feature in input_data:
                    input_array.append(np.log1p(input_data[base_feature]))
            else:
                input_array.append(input_data[feature])
        
        input_array = np.array(input_array).reshape(1, -1)
        
        # Scale the input
        scaled_input = scaler.transform(input_array)
        
        # Make prediction
        predicted_price = model.predict(scaled_input)[0]
        
        # Display result
        st.success("✅ Prediction Complete!")
        st.metric("Predicted Phone Price", f"${predicted_price:.2f}", delta=None)
        
        # Show input summary
        with st.expander("📊 Input Summary"):
            summary_df = pd.DataFrame({
                'Specification': list(input_data.keys()),
                'Value': list(input_data.values())
            })
            st.dataframe(summary_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.info("Please ensure all input values are valid numbers.")

st.markdown("---")
st.markdown("**Model Information:** Trained using Ridge/Lasso/ElasticNet regression with optimal hyperparameters")
import streamlit as st
import pandas as pd
import datetime
from ecommerce_testing import EcommerceTestingPipeline

# Initialize the testing pipeline
pipeline = EcommerceTestingPipeline()

# Streamlit Page Config
st.set_page_config(page_title="E-Commerce Marketing Optimization", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Model Prediction"])

# Home Page
if selection == "Home":
    st.title("E-Commerce Marketing Optimization Project")
    st.write("""
        ### Overview
        In the ever-evolving landscape of e-commerce, companies face the challenge of optimizing their marketing strategies to achieve maximum impact.
        This project focuses on optimizing the marketing budget for an e-commerce firm based in Ontario, Canada, which specializes in electronic products.
        The goal is to either streamline the marketing budget or strategically reallocate it across various marketing levers to enhance revenue response.

        ### Objectives
        - Develop Machine Learning for three product subcategories: camera accessory, home audio, and gaming accessory.
        - Create models for each subcategory to predict Gross Merchandise Value (GMV).
        - Use historical data, advertising spend, and other features to build predictive models.
    """)

# Model Prediction Page
elif selection == "Model Prediction":
    st.title("E-Commerce GMV Prediction")
    st.write("Enter the details below to predict the Gross Merchandise Value (GMV).")

    # Input Fields
    order_date = st.date_input("Order Date", min_value=datetime.date(2015, 7, 1), max_value=datetime.date(2016, 6, 30))
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
    sla_for_delivery = st.number_input("SLA for Delivery (days)", min_value=1, value=1)
    product_mrp = st.number_input("Product MRP", min_value=0.0, value=0.0)
    product_procurement_sla = st.number_input("Product Procurement SLA (days)", min_value=0, value=0)
    special_day_holiday = st.selectbox("Special Day Holiday", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    payment_type = st.selectbox("Payment Type", options=["Prepaid", "COD"])
    product_sub_category = st.selectbox("Product Sub-category", options=["CameraAccessory", "HomeAudio", "GamingAccessory"])
    product_analytic_vertical = st.text_input("Product Analytic Vertical")
    marketing_spend_in_cr = st.number_input("Marketing Spend in CR", min_value=0.0, value=0.0)

    if st.button("Predict GMV"):
        # Prepare the input data in a DataFrame
        input_data = pd.DataFrame([{
            'order_date': order_date,
            'Month': month,
            'sla': sla_for_delivery,
            'product_mrp': product_mrp,
            'product_procurement_sla': product_procurement_sla,
            'Special_day_holiday': special_day_holiday,
            's1_fact.order_payment_type': payment_type,
            'product_analytic_vertical': product_analytic_vertical,
            'product_analytic_sub_category': product_sub_category,
            'Marketing_spend_in_CR': marketing_spend_in_cr
        }])

        # Pass the input data to the testing pipeline for preprocessing and prediction
        prediction = pipeline.predict(input_data)

        # Display the prediction
        st.success(f"Predicted GMV: {prediction[0]:.2f}")

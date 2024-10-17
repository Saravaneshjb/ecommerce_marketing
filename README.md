# eCommerce Marketing Optimization Project

## Overview

The **eCommerce Marketing Optimization Project** aims to optimize marketing spend for an eCommerce firm specializing in electronic products, such as camera accessories, home audio, and gaming accessories. The project leverages data to build predictive Machine Learning (ML) and Deep Learning (DL) models that forecast Gross Merchandise Value (GMV) and help allocate the marketing budget more effectively. A Streamlit application is also provided for users to interact with the model and visualize predictions.

## Functionality

- **Data Preprocessing**: The project processes historical eCommerce sales data, including product information, marketing spends, and environmental factors. Data preprocessing includes feature engineering for paydays, holidays, and advertisement proportional investments.

- **Model Building**: Both ML and DL models are trained to predict GMV for three product subcategories (camera accessory, home audio, and gaming accessory). The models aim to provide weekly-level predictions for effective budget allocation.

- **Interactive Web Application**: A Streamlit app is built to allow users to input product information and marketing spends, and visualize the predicted GMV.

- **Proportional Allocation of Marketing Spend**: The project utilizes a proportional allocation approach to distribute monthly marketing spends across different product categories, enabling analysis of the effect of marketing investments on sales.

## Architecture

- **Frontend**: A Streamlit-based web application that allows users to input relevant data for GMV prediction, such as marketing spend, product type, delivery SLA, and others. It provides a simple and user-friendly interface to visualize results.

- **Backend**: Python scripts handle the data preprocessing, model training, and prediction. The backend is responsible for transforming user inputs, applying preprocessing steps, and making predictions using the trained model.

- **Model Training & Inference**: A Random Forest model is trained on historical sales data. The model artifacts and encoders are pickled for reusability in the prediction pipeline. The inference pipeline loads these artifacts to transform user inputs and generate predictions.

## Data Flow

1. **Data Preprocessing**: The dataset is preprocessed to handle missing values, encode categorical features, create new features, and calculate the proportional marketing spend. The training data covers July 2015 to June 2016.

2. **Model Training**: A Random Forest regressor is trained to predict the GMV for each product subcategory. Encoders for categorical features are also saved for use in the prediction pipeline.

3. **User Input & Prediction**: The Streamlit application accepts user inputs, preprocesses them, and predicts the GMV using the trained model. The model and preprocessing encoders are loaded from pickle files.

## Usage

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/username/ecommerce-marketing-optimization.git
   cd ecommerce-marketing-optimization
   ```

2. **Install Dependencies**
   Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Application**
   Start the Streamlit app using:
   ```bash
   streamlit run app.py
   ```

### Features for Prediction

The Streamlit application collects the following features from the user for GMV prediction:
- **Order Date**: The date of order placement (used to calculate marketing spend per day).
- **Month**: Month of the order.
- **SLA for Delivery**: Estimated number of days for delivery.
- **Product MRP**: Maximum retail price of the product.
- **Product Procurement SLA**: Estimated time taken to procure the product.
- **Special Day Holiday**: Whether the order was placed on a special holiday.
- **Payment Type**: Payment method (Card or COD).
- **Product Sub-category**: One of three subcategories - camera accessory, home audio, gaming accessory.
- **Product Analytic Vertical**: Detailed product type (e.g., Camera Tripod).
- **Marketing Spend in CR**: Total marketing spend for the given month.

### Model Training Workflow

- The **ecommerce_training.py** script performs data preprocessing, trains the model, and pickles the artifacts for later use.
- The **ecommerce_testing.py** script is responsible for using the trained model to make predictions based on new user inputs.

### Important Notes

- **Pickle Files**: All encoders and the model are saved as `.pkl` files to ensure consistency between training and prediction.
- **Data Preparation**: Proportional allocation of monthly marketing spends is calculated to match the level of granularity of the order-level dataset.

## Technologies Used

- **Python**: Data preprocessing, model building, and deployment.
- **Pandas & NumPy**: Data manipulation and preprocessing.
- **Scikit-learn**: Machine learning model training.
- **Streamlit**: Building an interactive frontend.

## Future Improvements

- **Additional Models**: Explore using deep learning models such as LSTM or Transformer-based models for time series prediction.
- **Feature Expansion**: Add more product attributes or customer features to improve model accuracy.
- **Model Deployment**: Deploy the model using cloud services for scalable access.



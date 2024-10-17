import pandas as pd
import pickle

class EcommerceTestingPipeline:
    def __init__(self):
        # Load the saved model and encoders
        self.model = pickle.load(open('D:\\Saravanesh Personal\\Guvi\\Guvi_Career_Fair\\Ecommerce_Marketing\\pickle_files\\random_forest_model.pkl', 'rb'))
        self.freq_encoding = pickle.load(open('D:\\Saravanesh Personal\\Guvi\\Guvi_Career_Fair\\Ecommerce_Marketing\\pickle_files\\freq_encoding.pkl', 'rb'))
        self.payment_type_encoder = pickle.load(open('D:\\Saravanesh Personal\\Guvi\\Guvi_Career_Fair\\Ecommerce_Marketing\\pickle_files\\payment_type_encoder.pkl', 'rb'))
        self.subcategory_encoder = pickle.load(open('D:\\Saravanesh Personal\\Guvi\\Guvi_Career_Fair\\Ecommerce_Marketing\\pickle_files\\product_subcategory_encoder.pkl', 'rb'))

    def preprocess_input(self, input_data):
        # Convert order_date to datetime and calculate days in month
        input_data['order_date'] = pd.to_datetime(input_data['order_date'])
        days_in_month = input_data['order_date'].dt.days_in_month.iloc[0]
        input_data['daily_proportional_investment_in_CR'] = input_data['Marketing_spend_in_CR'] / days_in_month

        # Apply frequency encoding to 'product_analytic_vertical'
        input_data['product_analytic_vertical_encoded'] = input_data['product_analytic_vertical'].map(self.freq_encoding)

        # One-Hot Encode 's1_fact.order_payment_type' using saved encoder
        for col in self.payment_type_encoder.columns:
            input_data[col] = 0
        if input_data['s1_fact.order_payment_type'].iloc[0] in self.payment_type_encoder.columns:
            input_data[input_data['s1_fact.order_payment_type'].iloc[0]] = 1

        # One-Hot Encode 'product_analytic_sub_category' using saved encoder
        for col in self.subcategory_encoder.columns:
            input_data[col] = 0
        if input_data['product_analytic_sub_category'].iloc[0] in self.subcategory_encoder.columns:
            input_data[input_data['product_analytic_sub_category'].iloc[0]] = 1

        # Drop original columns
        input_data.drop(['product_analytic_vertical', 's1_fact.order_payment_type', 'product_analytic_sub_category', 'Marketing_spend_in_CR', 'order_date'], axis=1, inplace=True)

        # Reorder columns to match the training data
        input_data = input_data[['Month', 'sla', 'product_mrp', 'product_procurement_sla',
                                 'Special_day_holiday', 'daily_proportional_investment_in_CR',
                                 'Prepaid', 'product_analytic_vertical_encoded', 'GamingAccessory', 'HomeAudio']]

        return input_data

    def predict(self, input_data):
        # Preprocess input data
        processed_data = self.preprocess_input(input_data)

        # Make prediction
        prediction = self.model.predict(processed_data)
        return prediction

# Example usage in testing pipeline
if __name__ == "__main__":
    # Assuming this is the input data
    input_data = pd.DataFrame({
        'order_date': ['2016-01-15'], 'Month': [1], 'sla': [3], 'product_mrp': [2000],
        'product_procurement_sla': [7], 'Special_day_holiday': [1],
        's1_fact.order_payment_type': ['Prepaid'], 'product_analytic_vertical': ['CameraTripod'],
        'product_analytic_sub_category': ['CameraAccessory'], 'Marketing_spend_in_CR': [100.0]
    })

    testing_pipeline = EcommerceTestingPipeline()
    prediction = testing_pipeline.predict(input_data)
    print(f"Predicted GMV: {prediction[0]}")

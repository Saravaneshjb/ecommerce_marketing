import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class EcommerceTrainingPipeline:
    def __init__(self, input_path):
        self.input_path = input_path
        self.df = None
        self.model = RandomForestRegressor(random_state=42)
        self.freq_encoding = None
        self.payment_type_encoder = None
        self.subcategory_encoder = None

    def load_data(self):
        print("Loading the data file to a dataframe")
        self.df = pd.read_csv(self.input_path, low_memory=False)

    def preprocess_data(self):
        # Filter for required sub-categories
        print('Starting the Data Preprocessing')
        self.df = self.df[self.df['product_analytic_sub_category'].isin(['CameraAccessory', 'HomeAudio', 'GamingAccessory'])].reset_index(drop=True)

        # Drop unwanted columns
        self.df.drop(columns=['fsn_id', 'order_id', 'order_item_id', 'deliverybdays', 'deliverycdays', 'cust_id', 'pincode'], inplace=True)

        # Remove duplicates
        self.df.drop_duplicates(inplace=True)

        # Handle missing values in `gmv`
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        self.df.dropna(subset=['gmv'], inplace=True)
        self.df['gmv'] = pd.to_numeric(self.df['gmv'], errors='coerce')

        # Drop rows with NaN in 'gmv' after conversion
        self.df.dropna(subset=['gmv'], inplace=True)

        # Create Pay_Day and Special_day_holiday features
        self.df['order_date'] = pd.to_datetime(self.df['order_date'], errors='coerce')
        self.df.dropna(subset=['order_date'], inplace=True)  # Drop rows with invalid dates
        self.df['Pay_Day'] = self.df['order_date'].dt.day.apply(lambda x: 1 if x in [1, 15] else 0)

        # Define special day ranges
        special_days = [
            ('2015-07-18', '2015-07-19'), ('2015-08-15', '2015-08-17'),
            ('2015-08-28', '2015-08-30'), ('2015-10-15', '2015-10-17'),
            ('2015-11-07', '2015-11-14'), ('2015-12-25', '2016-01-03'),
            ('2016-01-20', '2016-01-22'), ('2016-02-01', '2016-02-02'),
            ('2016-02-14', '2016-02-15'), ('2016-02-20', '2016-02-21'),
            ('2016-03-07', '2016-03-09'), ('2016-05-25', '2016-05-27')
        ]
        self.df['Special_day_holiday'] = 0
        for start_date, end_date in special_days:
            self.df.loc[(self.df['order_date'] >= pd.to_datetime(start_date)) & (self.df['order_date'] <= pd.to_datetime(end_date)), 'Special_day_holiday'] = 1

        # Filter for the required timeframe
        self.df = self.df[(self.df['order_date'] > '2015-06-30') & (self.df['order_date'] <= '2016-06-30')]

        # Include marketing data
        marketing_data = pd.DataFrame({
            'Year': [2015, 2015, 2015, 2015, 2015, 2015, 2016, 2016, 2016, 2016, 2016, 2016],
            'Month': [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6],
            'Total_Investment_in_CR': [17.1, 5.1, 96.3, 170.2, 51.2, 106.7, 74.2, 48.1, 100.0, 56.8, 78.1, 42.8]
        })
        self.df['Year'] = self.df['order_date'].dt.year
        self.df['Month'] = self.df['order_date'].dt.month
        monthly_gmv = self.df.groupby(['Year', 'Month', 'product_analytic_sub_category']).agg({'gmv': 'sum'}).reset_index()
        combined_data = pd.merge(monthly_gmv, marketing_data, how='left', on=['Year', 'Month'])
        combined_data['Total_GMV_Month'] = combined_data.groupby(['Year', 'Month'])['gmv'].transform('sum')
        combined_data['GMV_Proportion'] = combined_data['gmv'] / combined_data['Total_GMV_Month']
        combined_data['Proportional_Investment'] = combined_data['GMV_Proportion'] * combined_data['Total_Investment_in_CR']
        self.df = pd.merge(self.df, combined_data, how='left', on=['Year', 'Month', 'product_analytic_sub_category'])
        self.df['daily_proportional_investment_in_CR'] = self.df['Proportional_Investment'] / self.df['order_date'].dt.days_in_month

        # Drop unnecessary features
        self.df.drop(columns=['product_analytic_super_category', 'product_analytic_category', 'units', 'Pay_Day'], inplace=True)

        # Encode categorical features
        # One-Hot Encoding for 's1_fact.order_payment_type'
        self.payment_type_encoder = pd.get_dummies(self.df['s1_fact.order_payment_type'], drop_first=True, dtype=int)
        self.df = pd.concat([self.df, self.payment_type_encoder], axis=1)
        self.df.drop(columns='s1_fact.order_payment_type', inplace=True)

        # Frequency encoding for 'product_analytic_vertical'
        self.freq_encoding = self.df['product_analytic_vertical'].value_counts(normalize=True)
        self.df['product_analytic_vertical_encoded'] = self.df['product_analytic_vertical'].map(self.freq_encoding)
        self.df.drop(columns='product_analytic_vertical', inplace=True)

        # One-hot encode 'product_analytic_sub_category'
        self.subcategory_encoder = pd.get_dummies(self.df['product_analytic_sub_category'], drop_first=True, dtype=int)
        self.df = pd.concat([self.df, self.subcategory_encoder], axis=1)
        self.df.drop(columns='product_analytic_sub_category', inplace=True)
        print("All the preprocessing completed successfully")
        print("The columns in the self.df : ",self.df.columns)
        self.df=self.df[['Month', 'sla', 'product_mrp', 'product_procurement_sla',
       'Special_day_holiday', 'gmv_x', 'daily_proportional_investment_in_CR',
       'Prepaid', 'product_analytic_vertical_encoded','GamingAccessory', 'HomeAudio']]
    

    def train_model(self):
        # Ensure 'gmv' is still present in DataFrame
        print("Started the Model Training")
        if 'gmv_x' not in self.df.columns:
            raise KeyError("'gmv_x' not found in DataFrame after preprocessing")

        # # Drop 'order_date' column since it's a datetime and can't be used directly in model training
        # if 'order_date' in self.df.columns:
        #     self.df.drop(columns=['order_date'], inplace=True)       

        # Split data
        X = self.df.drop(columns=['gmv_x'])
        y = self.df['gmv_x']
        
        # Check for any remaining non-numeric data types
        # print(X.dtypes)  # Debug: Check if there are any non-numeric columns remaining

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)
        print("Completed MOdel Training")


    def save_artifacts(self):
        # Save model and encoders
        print("Started with the Pickling Process")
        pickle.dump(self.model, open('D:\\Saravanesh Personal\\Guvi\\Guvi_Career_Fair\\Ecommerce_Marketing\\pickle_files\\random_forest_model.pkl', 'wb'))
        pickle.dump(self.freq_encoding, open('D:\\Saravanesh Personal\\Guvi\\Guvi_Career_Fair\\Ecommerce_Marketing\\pickle_files\\freq_encoding.pkl', 'wb'))
        pickle.dump(self.payment_type_encoder, open('D:\\Saravanesh Personal\\Guvi\\Guvi_Career_Fair\\Ecommerce_Marketing\\pickle_files\\payment_type_encoder.pkl', 'wb'))
        pickle.dump(self.subcategory_encoder, open('D:\\Saravanesh Personal\\Guvi\\Guvi_Career_Fair\\Ecommerce_Marketing\\pickle_files\\product_subcategory_encoder.pkl', 'wb'))
        print("Completed the Pickling Process")

# Run the training pipeline
if __name__ == "__main__":
    training_pipeline = EcommerceTrainingPipeline("D:\\Saravanesh Personal\\Guvi\\Guvi_Career_Fair\\Ecommerce_Marketing\\Data\\ConsumerElectronics.csv")
    training_pipeline.load_data()
    training_pipeline.preprocess_data()
    training_pipeline.train_model()
    training_pipeline.save_artifacts()

import pandas as pd
import joblib
from datetime import datetime

scaler = joblib.load('/Users/junsookim/Desktop/self-learning/projects/AutoMaintenanceML/scaler.pkl')
model = joblib.load('/Users/junsookim/Desktop/self-learning/projects/AutoMaintenanceML/vehicle_maintenance_predictor.pkl')

# Function to preprocess dates into a numerical format (e.g., days since a reference date)
def preprocess_dates(df, date_columns):
    reference_date = datetime.now()
    for col in date_columns:
        df[col] = (pd.to_datetime(df[col]) - reference_date).dt.days
    return df

# Example of multiple new data inputs, more diverse including both likely and unlikely maintenance needs
data = {
    'Vehicle_Model': ['Truck', 'Car', 'Bus', 'SUV', 'Motorcycle'],
    'Mileage': [0, 30000, 189000, 50000, 1200],
    'Maintenance_History': ['Poor', 'Poor', 'Poor', 'Good', 'Excellent'],
    'Reported_Issues': [5, 2, 8, 1, 0],
    'Vehicle_Age': [3, 1, 10, 5, 2],
    'Fuel_Type': ['Diesel', 'Petrol', 'Diesel', 'Petrol', 'Electric'],
    'Transmission_Type': ['Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic'],
    'Engine_Size': [2500, 1800, 3000, 2200, 1100],
    'Odometer_Reading': [60000, 30000, 214125, 50000, 10000],
    'Last_Service_Date': ['2021-06-01', '2021-12-01', '2000-01-15', '2022-01-01', '2023-01-01'],
    'Warranty_Expiry_Date': ['2022-06-01', '2025-12-01', '2000-01-15', '2024-01-01', '2028-01-01'],
    'Owner_Type': ['First', 'Second', 'Third', 'First', 'First'],
    'Insurance_Premium': [300, 350, 250, 200, 180],
    'Service_History': [10, 15, 5, 8, 2],
    'Accident_History': [0, 1, 2, 0, 0],
    'Fuel_Efficiency': [15.5, 18.5, 12.0, 20.0, 50.0],
    'Tire_Condition': ['Good', 'Fair', 'Poor', 'New', 'New'],
    'Brake_Condition': ['New', 'Fair', 'Worn Out', 'Good', 'Excellent'],
    'Battery_Status': ['Good', 'Good', 'Poor', 'New', 'Excellent']
}

# Convert to DataFrame
new_data = pd.DataFrame(data)

# Preprocess date columns
date_cols = ['Last_Service_Date', 'Warranty_Expiry_Date']
new_data = preprocess_dates(new_data, date_cols)

# Ensure all categorical variables are encoded exactly as they were during training
new_data_encoded = pd.get_dummies(new_data)
# Align columns with the training data
missing_cols = set(scaler.feature_names_in_) - set(new_data_encoded.columns)
for c in missing_cols:
    new_data_encoded[c] = 0
new_data_encoded = new_data_encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale new data using the loaded scaler
new_data_prepared = scaler.transform(new_data_encoded)

# Predict maintenance needs using the loaded model
predictions = model.predict(new_data_prepared)

# Output the prediction
print("Predictions for maintenance needs:", predictions)
print("Does the car need maintenance? ", ["Yes" if pred == 1 else "No" for pred in predictions])

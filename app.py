from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS  

# add silly code for activating github actions

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

model = joblib.load('vehicle_maintenance_predictor.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/api/predict', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)  
    query = pd.get_dummies(query_df)
    query = query.reindex(columns=model_columns, fill_value=0)
    query_scaled = scaler.transform(query)
    prediction = model.predict(query_scaled)
    
    prediction = [int(pred) for pred in prediction]  
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

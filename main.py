import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('/Users/junsookim/Desktop/self-learning/projects/AutoMaintenanceML/vehicle_maintenance_data.csv')


numeric_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(exclude=['number']).columns


for col in numeric_cols:
    if data[col].isnull().any():
        data[col].fillna(data[col].median(), inplace=True)


for col in categorical_cols:
    if data[col].isnull().any():
        data[col].fillna(data[col].mode()[0], inplace=True)


y = data['Need_Maintenance']
X = data.drop(columns=['Need_Maintenance'])


X = pd.get_dummies(X, columns=categorical_cols)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 


joblib.dump(scaler, '/Users/junsookim/Desktop/self-learning/projects/AutoMaintenanceML/scaler.pkl')


joblib.dump(X.columns.tolist(), '/Users/junsookim/Desktop/self-learning/projects/AutoMaintenanceML/model_columns.pkl')


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')


model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Training Accuracy: {train_accuracy}')
print(f'Testing Accuracy: {test_accuracy}')

print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))


joblib.dump(model, '/Users/junsookim/Desktop/self-learning/projects/AutoMaintenanceML/vehicle_maintenance_predictor.pkl')

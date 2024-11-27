import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\ASIF MANZOOR\Downloads\Lohi\Sales Dataset.csv")  

# Preprocess data
X = data[['Product Price', 'Quantity Sold']]  # Features
y = data['Total Sales']  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Save model
joblib.dump(model, 'sales_prediction_model.pkl')
print("Model saved as 'sales_prediction_model.pkl'")

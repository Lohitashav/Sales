# **Real-Time Sales Prediction Dashboard**

## **Project Overview**
This project demonstrates a real-time data analytics dashboard using **Streamlit** integrated with a pre-trained **machine learning model** for predicting total sales. It mimics real-world scenarios by simulating e-commerce sales data in real-time, processes the data with a regression model, and visualizes predictions using interactive charts.

Key Features:
- Simulates real-time sales data (product type, price, quantity, etc.) with time-stamped entries.
- Utilizes a regression model to predict total sales.
- Displays interactive visualizations, including line charts for trends, bar charts for product-wise quantities, and histograms for price distribution.
- Provides user control over simulation parameters via a sidebar.



## **Setup and Execution Steps**

### **1. Prerequisites**
Ensure the following software/tools are installed:
- Python (>=3.8)
- Pip (comes with Python)
- Libraries: Streamlit, Pandas, Scikit-learn, Plotly, and Joblib

Install these dependencies by running:
```bash
pip install streamlit pandas scikit-learn plotly joblib
```

### **2. Clone the Repository**
Download or clone the project repository:
```bash
git clone <repository-link>
cd <repository-folder>
```

### **3. Prepare the Dataset**
Use any e-commerce dataset (e.g., from Kaggle) containing product prices, product types, and quantities sold. Clean and preprocess the data to create training data.

Example dataset format:
| Product Type  | Product Price | Quantity Sold |
|---------------|---------------|---------------|
| Electronics   | 89.99         | 5             |
| Clothing      | 29.99         | 3             |

Save this as `ecommerce_data.csv` in the project folder.

### **4. Train the Model**
Run the model training script:
```bash
python train_model.py
```

This script:
- Loads the cleaned dataset.
- Splits the data into training and testing sets.
- Trains a regression model using Scikit-learn.
- Saves the trained model as `sales_prediction_model.pkl`.

### **5. Run the Dashboard**
Launch the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

### **6. Interact with the Dashboard**
- **Start Simulation**: Generates real-time sales data.
- **View Predictions**: Displays predicted sales for the generated data.
- **Analyze Trends**: Explore visualizations for sales trends, product distribution, and pricing.


## **Description of the ML Model**

### **1. Objective**
The machine learning model predicts **total sales** based on:
- **Product Price**
- **Quantity Sold**

### **2. Model Training**
- **Algorithm**: Linear Regression (Scikit-learn)
- **Input Features**:
  - `Product Price`: Continuous numerical values representing the price of products.
  - `Quantity Sold`: Integer values indicating the number of units sold.
- **Output**:
  - `Total Sales`: Predicted as `Product Price × Quantity Sold`.

### **3. Model Performance**
The model was evaluated using standard regression metrics:
- **Mean Squared Error (MSE)**
- **R² Score**

Training and evaluation results are printed during the execution of `train_model.py`.

### **4. Saved Model**
The trained regression model is serialized and saved using **Joblib** as `sales_prediction_model.pkl`.


## **Visualizations**
- **Line Chart**: Displays sales predictions over time.
- **Bar Chart**: Shows total quantity sold per product type.
- **Histogram**: Analyzes the price distribution for different product categories.



## **Project Structure**
```
project-folder/
├── dashboard.py             # Streamlit app
├── train_model.py           # Model training script
├── sales_prediction_model.pkl  # Pre-trained model
├── ecommerce_data.csv       # Sample dataset (replace with your dataset)
├── README.md                # Project documentation
```



### **Dashboard Overview**
![Dashboard Screenshot](https://github.com/Lohitashav/Sales/raw/main/E-Commerce%20Sales/Dashboard/Screenshot%202024-11-27%20151902.png)

![Dashboard Screenshot](https://github.com/Lohitashav/Sales/raw/main/E-Commerce%20Sales/Dashboard/Screenshot%202024-11-27%20151917.png)


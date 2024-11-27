import streamlit as st
import pandas as pd
import joblib
from time import sleep
import random
import plotly.express as px

# Load the pre-trained model
model = joblib.load('sales_prediction_model.pkl')

# Simulate data
def simulate_data():
    product_types = ['Electronics', 'Clothing', 'Accessories']
    while True:
        data = {
            'timestamp': pd.Timestamp.now(),
            'Product Type': random.choice(product_types),
            'Product Price': random.uniform(10, 100),
            'Quantity Sold': random.randint(1, 10)
        }
        yield pd.DataFrame([data])

# Initialize a DataFrame to store all simulated data
all_data = pd.DataFrame()

# Streamlit app
st.title("Real-Time Sales Prediction Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Simulation Settings")
    delay = st.slider("Update Frequency (seconds)", 0.1, 5.0, 1.0)

# Real-time data display
st.header("Simulated Data")
data_placeholder = st.empty()

# Real-time predictions
st.header("Predicted Total Sales")
prediction_placeholder = st.empty()

# Visualizations
st.header("Data Trends")
line_chart_placeholder = st.empty()
bar_chart_placeholder = st.empty()
histogram_placeholder = st.empty()

# Start simulation
if st.button("Start Simulation"):
    for simulated_row in simulate_data():
        # Append new data to the global DataFrame
        all_data = pd.concat([all_data, simulated_row], ignore_index=True)

        # Display real-time data
        data_placeholder.write(simulated_row)

        # Predict total sales
        X = simulated_row[['Product Price', 'Quantity Sold']]
        prediction = model.predict(X)[0]
        prediction_placeholder.write(f"Predicted Total Sales: ${prediction:.2f}")

        # Line Chart: Total sales trend over time
        if len(all_data) > 1:
            # Predict total sales for all simulated data
            all_data['Predicted Sales'] = model.predict(all_data[['Product Price', 'Quantity Sold']])

            # Line Chart: Sales prediction trend over time
            line_fig = px.line(
                all_data,
                x='timestamp',
                y='Predicted Sales',
                title="Sales Prediction Over Time"
            )
            line_chart_placeholder.plotly_chart(line_fig, use_container_width=True)

            # Bar Chart: Total quantity sold by product type
            grouped_data = all_data.groupby('Product Type', as_index=False)[['Quantity Sold']].sum()
            bar_fig = px.bar(
                grouped_data,
                x='Product Type',
                y='Quantity Sold',
                title="Total Quantity Sold by Product Type",
                color='Product Type'
            )
            bar_chart_placeholder.plotly_chart(bar_fig, use_container_width=True)

            # Histogram: Product price distribution
            hist_fig = px.histogram(
                all_data,
                x='Product Price',
                nbins=10,
                title="Product Price Distribution",
                color='Product Type'
            )
            histogram_placeholder.plotly_chart(hist_fig, use_container_width=True)

        # Sleep for the specified delay
        sleep(delay)

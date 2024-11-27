import pandas as pd
import random
from time import sleep

# Load the e-commerce sales dataset
data = pd.read_csv(r"C:\Users\ASIF MANZOOR\Downloads\Lohi\Sales Dataset.csv")  

def simulate_data():
    while True:
        row = data.sample(1)  
        row['timestamp'] = pd.Timestamp.now()  
        yield row

if __name__ == "__main__":
    for simulated_row in simulate_data():
        print(simulated_row)
        sleep(1)  # Simulate a delay

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Load the dataset
data = pd.read_csv('power_consumption.csv')

# Convert 'Date' and 'Time' into datetime format
data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])

# Extract features from datetime (hour, day, month)
data['hour'] = data['datetime'].dt.hour
data['day'] = data['datetime'].dt.day
data['month'] = data['datetime'].dt.month

# Drop 'Date', 'Time', and 'datetime' columns
data.drop(columns=['Date', 'Time', 'datetime'], inplace=True)

# Define the features (X) and the target (y)
X = data[['hour', 'day', 'month', 'Global_reactive_power', 'Voltage']]
y = data['Global_active_power']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)



# Print the first 10 actual vs predicted values
print("\nFirst 10 Actual vs Predicted Values:")
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual:.3f}, Predicted: {predicted:.3f}")

# Predict future consumption based on new data
new_data = pd.DataFrame({
    'hour': [12, 13, 14],
    'day': [1, 1, 1],
    'month': [1, 1, 1],
    'Global_reactive_power': [0.1, 0.11, 0.09],
    'Voltage': [240.0, 241.0, 242.0]
})

future_consumption = model.predict(new_data)
print(f"\nPredicted Future Consumption: {future_consumption}")

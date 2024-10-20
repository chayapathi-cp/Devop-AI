# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Step 2: Load the dataset

data = pd.read_csv('house_data.csv')

# Step 3: Data Preprocessing
# Separate features (X) and target variable (y)
X = data[['size', 'bedrooms', 'age']]  # Features (you can modify based on your dataset)
y = data['price']  # Target variable

# Step 4: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create a Linear Regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = model.predict(X_test)

comparison = pd.DataFrame({'Actual Price': y_test.values, 'Predicted Price': y_pred})
pd.set_option('display.float_format', '{:.2f}'.format)

print("\nHouse Price Predictions vs Actual Prices:")
print(comparison.head())  # Display the first few predictions for comparison


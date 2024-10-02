# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# Sample dataset (income, loan_amount, credit_score, and default status)
# Format: [income, loan_amount, credit_score], default (1 if default, 0 if not)
data = np.array([
    [50000, 20000, 700, 0],
    [60000, 25000, 750, 0],
    [120000, 50000, 800, 0],
    [35000, 15000, 600, 1],
    [45000, 18000, 650, 1],
    [80000, 30000, 720, 0],
    [30000, 12000, 580, 1],
    [100000, 40000, 780, 0],
    [25000, 10000, 550, 1],
    [90000, 35000, 740, 0]
])

# Split the dataset into features (X) and labels (y)
X = data[:, :3]  # income, loan_amount, credit_score
y = data[:, 3]   # default status (0 or 1)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict the risk of default for a new customer
new_customer = np.array([[75000, 28000, 710]])  # Example customer data
default_prediction = model.predict(new_customer)
print("Default risk:", "Yes" if default_prediction == 1 else "No")


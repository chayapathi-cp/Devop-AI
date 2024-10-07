import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load sample data for testing
@pytest.fixture
def sample_data():
    # Sample data similar to what your 'data.csv' would have
    data = pd.DataFrame({
        'person_age': [25, 45, 35],
        'person_income': [50000, 120000, 75000],
        'person_home_ownership': ['RENT', 'OWN', 'RENT'],
        'person_emp_length': [2, 15, 10],
        'loan_intent': ['PERSONAL', 'EDUCATION', 'PERSONAL'],
        'loan_grade': ['B', 'A', 'C'],
        'loan_amnt': [15000, 30000, 28000],
        'loan_int_rate': [10.5, 7.2, 12.0],
        'loan_percent_income': [0.3, 0.25, 0.37],
        'cb_person_default_on_file': ['N', 'N', 'Y'],
        'cb_person_cred_hist_length': [5, 10, 7],
        'loan_status': [0, 1, 0]  # 0 for no default, 1 for default
    })
    return data

# Test the preprocessing (label encoding and scaling)
def test_preprocessing(sample_data):
    categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    label_encoders = {}
    
    # Encode categorical columns
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        sample_data[col] = label_encoders[col].fit_transform(sample_data[col])

    assert sample_data['person_home_ownership'].dtype == 'int32'
    assert sample_data['loan_intent'].dtype == 'int32'
    assert sample_data['loan_grade'].dtype == 'int32'
    assert sample_data['cb_person_default_on_file'].dtype == 'int32'

    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_columns = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 
                         'loan_percent_income', 'cb_person_cred_hist_length']
    sample_data[numerical_columns] = scaler.fit_transform(sample_data[numerical_columns])

    # Ensure that the data has been scaled
    assert np.isclose(sample_data['person_age'].mean(), 0, atol=1e-1)
    assert np.isclose(sample_data['loan_amnt'].std(), 1, atol=1e-1)

# Test training the model
def test_model_training(sample_data):
    # Preprocess the sample data
    categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    label_encoders = {}
    
    # Encode categorical columns
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        sample_data[col] = label_encoders[col].fit_transform(sample_data[col])

    # Define features (X) and target (y)
    X = sample_data[['person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
                     'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                     'cb_person_default_on_file', 'cb_person_cred_hist_length']]
    y = sample_data['loan_status']

    # Standardize numerical columns
    scaler = StandardScaler()
    X[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 
       'loan_percent_income', 'cb_person_cred_hist_length']] = scaler.fit_transform(
        X[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 
           'loan_percent_income', 'cb_person_cred_hist_length']]
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Ensure that the model has been trained
    assert model.coef_ is not None
    assert model.intercept_ is not None

# Test model predictions
def test_model_prediction(sample_data):
    # Similar preprocessing as in the model training function
    categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    label_encoders = {}
    
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        sample_data[col] = label_encoders[col].fit_transform(sample_data[col])

    X = sample_data[['person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
                     'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                     'cb_person_default_on_file', 'cb_person_cred_hist_length']]
    y = sample_data['loan_status']

    scaler = StandardScaler()
    X[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 
       'loan_percent_income', 'cb_person_cred_hist_length']] = scaler.fit_transform(
        X[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 
           'loan_percent_income', 'cb_person_cred_hist_length']]
    )

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Create a new customer data point
    new_customer = np.array([[35, 75000, 1, 10, 2, 1, 28000, 12, 0.25, 0, 7]])
    new_customer[:, [0, 1, 3, 6, 7, 8, 10]] = scaler.transform(new_customer[:, [0, 1, 3, 6, 7, 8, 10]])

    # Make prediction
    prediction = model.predict(new_customer)
    
    # Check that the prediction is valid (0 or 1)
    assert prediction in [0, 1]



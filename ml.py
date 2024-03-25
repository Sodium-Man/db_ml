# Importing necessary libraries
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Initialize Flask app
app = Flask(__name__)

# Read the dataset
data = pd.read_csv("data.csv")

# Split the data into features and target variable
X = data.drop('alert', axis=1)
y = data['alert']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
classifier = LogisticRegression()

# Train the model
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Function to take user input and predict
def predict_alert(mq6, mq7, mq9, mq135):
    # Prepare input as a 2D array (required by predict() method)
    input_data = np.array([[mq6, mq7, mq9, mq135]])

    # Make prediction
    prediction = classifier.predict(input_data)

    # Interpret prediction
    if prediction[0] == 0:
        return "No alert"
    elif prediction[0] == 1:
        return "Alert"

# Define route for home page
@app.route('/')
def home():
    return render_template('index2.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        mq6 = float(request.form['mq6'])
        mq7 = float(request.form['mq7'])
        mq9 = float(request.form['mq9'])
        mq135 = float(request.form['mq135'])

        prediction = predict_alert(mq6, mq7, mq9, mq135)
        return render_template('index2.html', prediction=prediction)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

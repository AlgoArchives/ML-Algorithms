import numpy as np
from sklearn.linear_model import LinearRegression

def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, X_new):
    y_pred = model.predict(X_new)
    return y_pred

def main():
    # Generate some example data for training
    np.random.seed(42)
    X_train = 2 * np.random.rand(100, 1)
    y_train = 4 + 3 * X_train + np.random.randn(100, 1) # y = 4 + 3x + Gaussian noise ------------- change wi and b values to see how they affect the model

    # Train the linear regression model
    model = train_linear_regression(X_train, y_train)

    # Get input for prediction
    try:
        new_input = float(input("Enter a new X value for prediction: "))
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return

    X_new = np.array([[new_input]])

    # Make prediction
    prediction = predict(model, X_new)

    # Display the prediction
    print(f"The predicted y value for X = {new_input} is: {prediction[0][0]}")

if __name__ == "__main__":
    main()
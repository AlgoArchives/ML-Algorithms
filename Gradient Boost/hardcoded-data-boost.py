import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Create a DataFrame with hardcoded data
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [10, 20, 30, 40, 50],
    'Target': [15, 25, 35, 45, 55]
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('hardcoded_data.csv', index=False)

# Load the CSV file
df = pd.read_csv('hardcoded_data.csv')

# Splitting the data into features and target
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and fitting the gradient boosting model
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_regressor.fit(X_train, y_train)

# Making predictions on the test set
y_pred = gb_regressor.predict(X_test)

# Calculating the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
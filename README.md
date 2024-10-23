
# Mileage-prediction-code 
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv('your_dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Features and target variable
# Assuming columns 'engine_size', 'weight', 'horsepower' as features and 'mileage' as the target
X = data[['engine_size', 'weight', 'horsepower']]  # Independent variables (features)
y = data['mileage']  # Dependent variable (target)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions using the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plotting the true vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('True Mileage')
plt.ylabel('Predicted Mileage')
plt.title('True vs Predicted Mileage')
plt.show()

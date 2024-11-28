# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Generate random dataset  dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# Visualize the data scatter in blue color




# Visualize the data 
plt.scatter(X, y, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter plot of the generated data')
plt.show()



# Split the dataset into training and testing sets 20,80 approuch
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



## 3. Training the Linear Regression Model
print("Training set X:", X_train.shape)
print("Training set y:", y_train.shape)
print("Test set X:", X_test.shape)
print("Test set y:", y_test.shape)



# Create a Linear Regression model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)

# Model coefficients
intercept = model.intercept_[0]
coefficient = model.coef_[0][0]

print(f"Intercept: {intercept}")
print(f"Coefficient: {coefficient}")



## 4. Making Predictions on the Test Set
y_pred = model.predict(X_test)

# Step 6: Print the predictions
print("Predictions on the test set:")
print(y_pred)




# Optional: Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Predictions')
plt.legend()
plt.show()




# Step 6: Evaluate the model - Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)
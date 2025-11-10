import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Create dummy dataset
# Advertising budget in TV, Radio, Newspaper vs Sales
data = {
    'TV': [230, 44, 17, 151, 180, 8, 57, 120, 77, 90],
    'Radio': [37, 39, 45, 41, 10, 48, 32, 19, 23, 5],
    'Newspaper': [69, 45, 69, 58, 52, 75, 23, 24, 31, 60],
    'Sales': [22, 10, 9, 18, 12, 7, 11, 15, 12, 14]
}
df = pd.DataFrame(data)

# Step 2: Features and target
X = df[['TV','Radio','Newspaper']]
y = df['Sales']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluate
print("R2 Score:", r2_score(y_test, y_pred))

# Step 7: Visualization
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Predicted vs Actual Sales")
plt.show()# Sales-Prediction-using-Linear-Regression

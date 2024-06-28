import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# File path (adjust as needed)
file_path = 'updated_dating.csv'

# Load dataset
df = pd.read_csv(file_path)
print("Dataset loaded successfully!")

# Display first rows
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Fill missing values in numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Check again for missing values
print("Missing values after filling:")
print(df.isnull().sum())

# Features and target variable
X = df[['attr', 'fun']].values  # Only select 'attr' and 'fun' for simplicity in plotting
y = df['dec']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features standardized!")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets!")

# Train the SVM model with RBF kernel and higher C parameter
svm = SVC(kernel='rbf', C=1000, gamma=1)
svm.fit(X_train, y_train)
print("SVM model trained!")

# Predict on the test set
y_pred = svm.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Create a meshgrid for decision boundary plotting with a reduced step size
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict using the SVM model
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.6)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('Standardized Attractiveness')
plt.ylabel('Standardized Fun')
plt.title('Decision Boundary for Attractiveness and Fun (RBF Kernel)')
plt.colorbar()
plt.show()


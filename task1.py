import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
file_path = '/Users/pragatik/Desktop/projectXfolder/projectX/updated_dating.csv'  # Ensure the file path is correct
df = pd.read_csv(file_path)

# Display first rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing values in each column
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Check again for missing values
print(df.isnull().sum())

# Features and target variable
X = df[['attr', 'sinc', 'intel', 'fun', 'amb', 'like', 'gender']]
y = df['dec']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the SVM model
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}

# Perform grid search
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters
print('Best parameters:', grid_search.best_params_)

# Retrain SVM with best parameters
best_svm = grid_search.best_estimator_
best_svm.fit(X_train, y_train)

# Select the 'attr' and 'fun' features for plotting
X_plot = df[['attr', 'fun']].values
y_plot = df['dec'].values

# Standardize the selected features
scaler_plot = StandardScaler()
X_plot_scaled = scaler_plot.fit_transform(X_plot)

# Create a grid to plot decision boundary
x_min, x_max = X_plot_scaled[:, 0].min() - 1, X_plot_scaled[:, 0].max() + 1
y_min, y_max = X_plot_scaled[:, 1].min() - 1, X_plot_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict using the best SVM model
Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel(), 
                           np.mean(X_train[:, 2])*np.ones_like(xx.ravel()), 
                           np.mean(X_train[:, 3])*np.ones_like(xx.ravel()), 
                           np.mean(X_train[:, 4])*np.ones_like(xx.ravel()), 
                           np.mean(X_train[:, 5])*np.ones_like(xx.ravel()), 
                           np.mean(X_train[:, 6])*np.ones_like(xx.ravel())])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X_plot_scaled[:, 0], X_plot_scaled[:, 1], c=y_plot, cmap='coolwarm', edgecolors='k')
plt.xlabel('Attractiveness')
plt.ylabel('Fun')
plt.title('Decision Boundary for Attractiveness and Fun')
plt.show()



![Renu Sankhla](https://miro.medium.com/v2/resize:fill:64:64/1*kJzzg0ZAa3-h7MZhT56VtQ.png)

As AI continues to boom, understanding Machine Learning (ML) — the foundational layer of AI — has become essential. Machine Learning is a subset of Artificial Intelligence (AI) where algorithmic models are trained on data to make accurate predictions or decisions without being explicitly programmed.

Key Branches of Machine Learning

- **Supervised Learning** — The model learns from labeled data (input-output pairs).
- **Unsupervised Learning** — The model finds patterns in unlabeled data.
- **Reinforcement Learning** — The model learns by interacting with an environment and receiving rewards/penalties.
- **Semi-Supervised Learning** — Combines a small amount of labeled data with a large amount of unlabelled data.
- **Self-Supervised Learning** — A form of unsupervised learning where the model generates its own labels from the data.

— — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —

# **Supervised Learning :**

It is a branch of Machine Learning (ML) where the model is trained on labeled data — meaning the input data is paired with the correct output. The goal is for the model to learn a mapping function from inputs to outputs so that it can make accurate predictions on new, unseen data.

Supervised Learning is broadly categorized into two types:

- Classification : Used when the output is a category or class label.
- Regression : Used when the output is a continuous value.

**Classification Algorithms :**

- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Neural Networks

**Regression Algorithms :**

- Linear Regression
- Polynomial Regression
- Ridge & Lasso Regression
- Decision Trees (for regression)

# **1. Logistic Regression**

Logistic Regression uses the logistic function to model the probability of class membership. Despite its name, it’s a classification algorithm that’s particularly effective for binary classification problems.

**Key Features:**

- Uses sigmoid function to map predictions to probabilities
- Assumes linear relationship between features and log-odds
- Provides interpretable coefficients
- Works well with linearly separable data

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                         n_informative=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

# **2. Decision Trees**

Decision Trees create a model that predicts target values by learning simple decision rules inferred from data features. They’re highly interpretable and can handle both numerical and categorical data.

**Key Features:**

- Easy to understand and interpret
- Requires little data preparation
- Can handle both numerical and categorical features
- Prone to overfitting with deep trees

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Create and train the model
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")

# Visualize the tree (first few levels)
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, max_depth=3, filled=True, feature_names=['Feature 1', 'Feature 2'])
plt.title("Decision Tree Visualization")
plt.show()
```

# **3. Random Forest**

Random Forest combines multiple decision trees to create a more robust and accurate model. It uses ensemble learning to reduce overfitting and improve generalization.

**Key Features:**

- Reduces overfitting compared to single decision trees
- Handles large datasets efficiently
- Provides feature importance rankings
- Works well with default parameters

```python
from sklearn.ensemble import RandomForestClassifier

# Create and train the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# Feature importance
feature_importance = rf_classifier.feature_importances_
print(f"Feature Importance: {feature_importance}")

# Plot feature importance
plt.figure(figsize=(8, 6))
plt.bar(['Feature 1', 'Feature 2'], feature_importance)
plt.title("Feature Importance in Random Forest")
plt.ylabel("Importance")
plt.show()
```

# **4. Support Vector Machines (SVM)**

SVM finds the optimal hyperplane that separates classes with maximum margin. It’s effective for high-dimensional data and can handle non-linear relationships using kernel functions.

**Key Features:**

- Effective in high-dimensional spaces
- Memory efficient (uses subset of training points)
- Versatile with different kernel functions
- Works well with clear margin separation

```python
from sklearn.svm import SVC

# Create and train the model
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.4f}")

# Try different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svm_temp = SVC(kernel=kernel, random_state=42)
    svm_temp.fit(X_train, y_train)
    y_pred_temp = svm_temp.predict(X_test)
    accuracy_temp = accuracy_score(y_test, y_pred_temp)
    print(f"SVM with {kernel} kernel: {accuracy_temp:.4f}")
```

# **Regression Algorithms**

Regression algorithms predict continuous numerical values.

# **1. Linear Regression**

Linear Regression models the relationship between features and target variable using a linear equation. It’s the foundation of many other regression techniques.

**Key Features:**

- Simple and interpretable
- Assumes linear relationship
- Sensitive to outliers
- Requires feature scaling for optimal performance

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

# Split the data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# Create and train the model
linear_reg = LinearRegression()
linear_reg.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_linear = linear_reg.predict(X_test_reg)

# Evaluate the model
mse_linear = mean_squared_error(y_test_reg, y_pred_linear)
r2_linear = r2_score(y_test_reg, y_pred_linear)
print(f"Linear Regression MSE: {mse_linear:.4f}")
print(f"Linear Regression R²: {r2_linear:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test_reg, y_test_reg, alpha=0.5, label='Actual')
plt.scatter(X_test_reg, y_pred_linear, alpha=0.5, label='Predicted')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.show()
```

# **2. Polynomial Regression**

Polynomial Regression extends linear regression by using polynomial features to capture non-linear relationships between variables.

**Key Features:**

- Captures non-linear relationships
- Can overfit with high-degree polynomials
- Requires careful degree selection
- Good for curved relationships

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create polynomial regression pipeline
poly_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])

# Train the model
poly_reg.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_poly = poly_reg.predict(X_test_reg)

# Evaluate the model
mse_poly = mean_squared_error(y_test_reg, y_pred_poly)
r2_poly = r2_score(y_test_reg, y_pred_poly)

print(f"Polynomial Regression MSE: {mse_poly:.4f}")
print(f"Polynomial Regression R²: {r2_poly:.4f}")

# Compare different degrees
degrees = [1, 2, 3, 4, 5]
mse_scores = []

for degree in degrees:
    poly_temp = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    poly_temp.fit(X_train_reg, y_train_reg)
    y_pred_temp = poly_temp.predict(X_test_reg)
    mse_temp = mean_squared_error(y_test_reg, y_pred_temp)
    mse_scores.append(mse_temp)

plt.figure(figsize=(10, 6))
plt.plot(degrees, mse_scores, marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs Polynomial Degree")
plt.show()
```

# **3. Ridge & Lasso Regression**

Ridge and Lasso regression add regularization terms to prevent overfitting by penalizing large coefficients.

**Ridge Regression (L2 Regularization):**

- Shrinks coefficients towards zero
- Keeps all features
- Good when all features are relevant

**Lasso Regression (L1 Regularization):**

- Can set coefficients to exactly zero
- Performs feature selection
- Good for sparse models

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Generate multi-feature data
X_multi, y_multi = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_multi)
X_test_scaled = scaler.transform(X_test_multi)

# Ridge Regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_scaled, y_train_multi)
y_pred_ridge = ridge_reg.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test_multi, y_pred_ridge)
r2_ridge = r2_score(y_test_multi, y_pred_ridge)

print(f"Ridge Regression MSE: {mse_ridge:.4f}")
print(f"Ridge Regression R²: {r2_ridge:.4f}")

# Lasso Regression
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X_train_scaled, y_train_multi)
y_pred_lasso = lasso_reg.predict(X_test_scaled)

mse_lasso = mean_squared_error(y_test_multi, y_pred_lasso)
r2_lasso = r2_score(y_test_multi, y_pred_lasso)

print(f"Lasso Regression MSE: {mse_lasso:.4f}")
print(f"Lasso Regression R²: {r2_lasso:.4f}")

# Compare coefficients
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(range(len(linear_reg.coef_)), linear_reg.coef_)
plt.title("Linear Regression Coefficients")
plt.xlabel("Feature")
plt.ylabel("Coefficient")

plt.subplot(1, 3, 2)
plt.bar(range(len(ridge_reg.coef_)), ridge_reg.coef_)
plt.title("Ridge Regression Coefficients")
plt.xlabel("Feature")
plt.ylabel("Coefficient")

plt.subplot(1, 3, 3)
plt.bar(range(len(lasso_reg.coef_)), lasso_reg.coef_)
plt.title("Lasso Regression Coefficients")
plt.xlabel("Feature")
plt.ylabel("Coefficient")

plt.tight_layout()
plt.show()

# Feature selection with Lasso
selected_features = np.where(lasso_reg.coef_ != 0)[0]
print(f"Lasso selected {len(selected_features)} features out of {len(lasso_reg.coef_)}")
```

# **4. Decision Trees (for Regression)**

Decision Trees can also be used for regression by predicting continuous values instead of class labels.

**Key Features:**

- Non-parametric method
- Handles non-linear relationships
- Easy to interpret
- Can overfit without proper pruning

```python
from sklearn.tree import DecisionTreeRegressor

# Create and train the model
dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_regressor.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_dt_reg = dt_regressor.predict(X_test_reg)

# Evaluate the model
mse_dt_reg = mean_squared_error(y_test_reg, y_pred_dt_reg)
r2_dt_reg = r2_score(y_test_reg, y_pred_dt_reg)

print(f"Decision Tree Regression MSE: {mse_dt_reg:.4f}")
print(f"Decision Tree Regression R²: {r2_dt_reg:.4f}")

# Compare different max_depth values
depths = [1, 3, 5, 7, 10, None]
mse_scores_dt = []

for depth in depths:
    dt_temp = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt_temp.fit(X_train_reg, y_train_reg)
    y_pred_temp = dt_temp.predict(X_test_reg)
    mse_temp = mean_squared_error(y_test_reg, y_pred_temp)
    mse_scores_dt.append(mse_temp)

plt.figure(figsize=(10, 6))
plt.plot(range(len(depths)), mse_scores_dt, marker='o')
plt.xlabel("Max Depth Index")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs Max Depth for Decision Tree Regression")
plt.xticks(range(len(depths)), [str(d) for d in depths])
plt.show()
```

# **Evaluation Metrics**

Evaluation metrics are essential for assessing model performance and making informed decisions about model selection and improvement.

- **Accuracy**: Correct predictions divided by total predictions.
- **Precision**: True positives / (True positives + False positives).
- **Recall**: True positives / (True positives + False negatives).
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC AUC**: Model’s ability to distinguish classes (1 = perfect).
- **MSE (Regression)**: Average squared difference between predicted and actual values.
- **R² (Regression)**: Proportion of variance explained by the model.

# **Classification Metrics**

## **1. Accuracy-Based Metrics**

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, matthews_corrcoef, cohen_kappa_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample classification data
X_class, y_class = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                      n_redundant=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train a sample model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Basic Classification Metrics
print("=== CLASSIFICATION METRICS ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

## **2. Confusion Matrix and Advanced Metrics**

```python
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Advanced Metrics
print(f"\nMatthews Correlation Coefficient: {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")

# Detailed Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
```

## **3. ROC Curve and Precision-Recall Curve**

```python
# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)

# Plot both curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ROC Curve
ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')
ax1.legend(loc="lower right")

# Precision-Recall Curve
ax2.plot(recall, precision, color='blue', lw=2, label='Precision-Recall Curve')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend()

plt.tight_layout()
plt.show()
```

## **4. Multi-class Classification Metrics**

```python
# Generate multi-class data
X_multi, y_multi = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                      n_redundant=5, n_classes=3, random_state=42)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42)

# Train model
model_multi = LogisticRegression(random_state=42)
model_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = model_multi.predict(X_test_multi)

# Multi-class metrics
print("=== MULTI-CLASS CLASSIFICATION METRICS ===")
print(f"Accuracy: {accuracy_score(y_test_multi, y_pred_multi):.4f}")
print(f"Precision (Macro): {precision_score(y_test_multi, y_pred_multi, average='macro'):.4f}")
print(f"Precision (Micro): {precision_score(y_test_multi, y_pred_multi, average='micro'):.4f}")
print(f"Precision (Weighted): {precision_score(y_test_multi, y_pred_multi, average='weighted'):.4f}")
print(f"Recall (Macro): {recall_score(y_test_multi, y_pred_multi, average='macro'):.4f}")
print(f"F1-Score (Macro): {f1_score(y_test_multi, y_pred_multi, average='macro'):.4f}")
```

# **Regression Metrics**

## **1. Basic Regression Metrics**

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_squared_log_error, median_absolute_error
)
from sklearn.linear_model import LinearRegression

# Generate regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# Train model
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

print("=== REGRESSION METRICS ===")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_reg, y_pred_reg):.4f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.4f}")
print(f"Median Absolute Error: {median_absolute_error(y_test_reg, y_pred_reg):.4f}")
```

## **2. Advanced Regression Metrics**

```python
# Custom regression metrics
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def adjusted_r2_score(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adjusted_r2

# Calculate advanced metrics
mape = mean_absolute_percentage_error(y_test_reg, y_pred_reg)
adj_r2 = adjusted_r2_score(y_test_reg, y_pred_reg, X_test_reg.shape[1])

print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
print(f"Adjusted R²: {adj_r2:.4f}")

# Residual Analysis
residuals = y_test_reg - y_pred_reg

plt.figure(figsize=(15, 5))

# Residual plot
plt.subplot(1, 3, 1)
plt.scatter(y_pred_reg, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Q-Q plot
from scipy import stats
plt.subplot(1, 3, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')

# Histogram of residuals
plt.subplot(1, 3, 3)
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')

plt.tight_layout()
plt.show()
```

# **Model Parameters vs Hyperparameters**

Understanding the distinction between parameters and hyperparameters is crucial for effective model development.

# **Parameters vs Hyperparameters**

```python
# Example: Linear Regression
from sklearn.linear_model import LinearRegression

# Create and fit model
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)

print("=== PARAMETERS vs HYPERPARAMETERS ===")
print("\nPARAMETERS (Learned during training):")
print(f"Coefficients (weights): {lr.coef_[:5]}...")  # Show first 5
print(f"Intercept (bias): {lr.intercept_:.4f}")

print("\nHYPERPARAMETERS (Set before training):")
print(f"fit_intercept: {lr.fit_intercept}")
print(f"copy_X: {lr.copy_X}")
print(f"n_jobs: {lr.n_jobs}")
print(f"positive: {lr.positive}")
```

**Common Hyperparameters by Algorithm**

```python
# Algorithm-specific hyperparameters
algorithms_hyperparameters = {
    "Logistic Regression": {
        "C": "Regularization strength (inverse)",
        "penalty": "Type of regularization (l1, l2, elasticnet)",
        "solver": "Algorithm to use in optimization",
        "max_iter": "Maximum number of iterations",
        "tol": "Tolerance for stopping criteria"
    },
    "Random Forest": {
        "n_estimators": "Number of trees in forest",
        "max_depth": "Maximum depth of trees",
        "min_samples_split": "Minimum samples required to split node",
        "min_samples_leaf": "Minimum samples required at leaf node",
        "max_features": "Number of features to consider for best split"
    },
    "SVM": {
        "C": "Regularization parameter",
        "kernel": "Kernel type (linear, poly, rbf, sigmoid)",
        "gamma": "Kernel coefficient for rbf, poly, sigmoid",
        "degree": "Degree of polynomial kernel",
        "tol": "Tolerance for stopping criterion"
    }
}

for algo, params in algorithms_hyperparameters.items():
    print(f"\n{algo} Hyperparameters:")
    for param, description in params.items():
        print(f"  {param}: {description}")
```

# **Hyperparameter Tuning Techniques**

# **1. Grid Search**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Create model
rf = RandomForestClassifier(random_state=42)
# Grid search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
# Fit grid search
print("=== GRID SEARCH ===")
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"Best estimator: {grid_search.best_estimator_}")
```

# **2. Random Search**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}
# Random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
print("\n=== RANDOM SEARCH ===")
random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")
```

# **3. Bayesian Optimization (using scikit-optimize)**

```python
# Note: This requires scikit-optimize: pip install scikit-optimize
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical

    # Define search space
    search_space = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 15),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical(['sqrt', 'log2'])
    }

    # Bayesian optimization
    bayes_search = BayesSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        search_spaces=search_space,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    print("\n=== BAYESIAN OPTIMIZATION ===")
    bayes_search.fit(X_train, y_train)

    print(f"Best parameters: {bayes_search.best_params_}")
    print(f"Best cross-validation score: {bayes_search.best_score_:.4f}")

except ImportError:
    print("\n=== BAYESIAN OPTIMIZATION ===")
    print("scikit-optimize not installed. Install with: pip install scikit-optimize")
```

# **4. Cross-Validation Strategies**

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, ShuffleSplit
)

# Different CV strategies
cv_strategies = {
    'KFold': KFold(n_splits=5, shuffle=True, random_state=42),
    'StratifiedKFold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'ShuffleSplit': ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
}
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("\n=== CROSS-VALIDATION STRATEGIES ===")
for cv_name, cv_strategy in cv_strategies.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
    print(f"{cv_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

# **Algorithm-Specific Parameters**

- Logistic Regression: `C` (regularization strength), `penalty` (l1/l2).
- Random Forest: `n_estimators` (number of trees), `max_depth` (tree depth).
- SVM: `C` (regularization), `kernel` (linear/rbf), `gamma` (kernel width).

# **Logistic Regression Parameters**

```python
from sklearn.linear_model import LogisticRegression

# Comprehensive parameter exploration
logistic_params = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'saga'],
    'max_iter': [100, 500, 1000]
}
# Grid search for logistic regression
lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    logistic_params,
    cv=5,
    scoring='accuracy'
)
print("=== LOGISTIC REGRESSION PARAMETER TUNING ===")
lr_grid.fit(X_train, y_train)
print(f"Best parameters: {lr_grid.best_params_}")
print(f"Best score: {lr_grid.best_score_:.4f}")
```

# **Random Forest Parameters**

```python
# Detailed Random Forest parameter analysis
rf_detailed = RandomForestClassifier(random_state=42)

# Parameter importance study
param_study = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}
print("\n=== RANDOM FOREST PARAMETER ANALYSIS ===")

# Study effect of n_estimators
n_estimators_scores = []
for n_est in param_study['n_estimators']:
    rf_temp = RandomForestClassifier(n_estimators=n_est, random_state=42)
    scores = cross_val_score(rf_temp, X_train, y_train, cv=5)
    n_estimators_scores.append(scores.mean())

# Plot parameter effects
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(param_study['n_estimators'], n_estimators_scores, marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('Cross-validation Score')
plt.title('Effect of n_estimators')

# Study effect of max_depth
max_depth_scores = []
for depth in param_study['max_depth']:
    rf_temp = RandomForestClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(rf_temp, X_train, y_train, cv=5)
    max_depth_scores.append(scores.mean())
plt.subplot(2, 2, 2)
plt.plot(range(len(param_study['max_depth'])), max_depth_scores, marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Cross-validation Score')
plt.title('Effect of max_depth')
plt.xticks(range(len(param_study['max_depth'])),
           [str(d) for d in param_study['max_depth']])

# Study effect of min_samples_split
min_samples_split_scores = []
for min_split in param_study['min_samples_split']:
    rf_temp = RandomForestClassifier(min_samples_split=min_split, random_state=42)
    scores = cross_val_score(rf_temp, X_train, y_train, cv=5)
    min_samples_split_scores.append(scores.mean())

plt.subplot(2, 2, 3)
plt.plot(param_study['min_samples_split'], min_samples_split_scores, marker='o')
plt.xlabel('Min Samples Split')
plt.ylabel('Cross-validation Score')
plt.title('Effect of min_samples_split')

# Study effect of min_samples_leaf
min_samples_leaf_scores = []
for min_leaf in param_study['min_samples_leaf']:
    rf_temp = RandomForestClassifier(min_samples_leaf=min_leaf, random_state=42)
    scores = cross_val_score(rf_temp, X_train, y_train, cv=5)
    min_samples_leaf_scores.append(scores.mean())
plt.subplot(2, 2, 4)
plt.plot(param_study['min_samples_leaf'], min_samples_leaf_scores, marker='o')
plt.xlabel('Min Samples Leaf')
plt.ylabel('Cross-validation Score')
plt.title('Effect of min_samples_leaf')
plt.tight_layout()
plt.show()
```

# **SVM Parameters**

```python
from sklearn.svm import SVC

# SVM parameter exploration
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}
print("\n=== SVM PARAMETER TUNING ===")
svm_grid = GridSearchCV(
    SVC(random_state=42),
    svm_params,
    cv=5,
    scoring='accuracy'
)
svm_grid.fit(X_train, y_train)
print(f"Best parameters: {svm_grid.best_params_}")
print(f"Best score: {svm_grid.best_score_:.4f}")
```

# **Best Practices and Guidelines**

# **1. Model Selection and Validation**

```python
from sklearn.model_selection import validation_curve, learning_curve

# Learning curves
def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Example learning curve
rf_best = RandomForestClassifier(**grid_search.best_params_, random_state=42)
plot_learning_curve(rf_best, X_train, y_train, "Random Forest Learning Curve")
```

# **2. Validation Curves**

```python
# Validation curve for a specific parameter
param_range = [10, 50, 100, 200, 500]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42), X_train, y_train,
    param_name='n_estimators', param_range=param_range,
    cv=5, scoring='accuracy', n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve for Random Forest n_estimators')
plt.legend()
plt.grid(True)
plt.show()
```

# **3. Feature Importance and Selection**

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Feature importance with Random Forest
rf_feature_importance = RandomForestClassifier(n_estimators=100, random_state=42)
rf_feature_importance.fit(X_train, y_train)

# Plot feature importance
feature_importance = rf_feature_importance.feature_importances_
feature_indices = np.argsort(feature_importance)[::-1]
plt.figure(figsize=(12, 8))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(min(20, len(feature_importance))), feature_importance[feature_indices[:20]])
plt.xticks(range(min(20, len(feature_importance))),
           [f"Feature {i}" for i in feature_indices[:20]], rotation=45)
plt.tight_layout()
plt.show()

# Recursive Feature Elimination
rfe = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=10)
rfe.fit(X_train, y_train)
print(f"\nSelected features (RFE): {np.where(rfe.support_)[0]}")
print(f"Feature ranking: {rfe.ranking_[:10]}...")
```

# **4. Model Comparison Framework**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Compare multiple algorithms
algorithms = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}
# Cross-validation comparison
results = {}
for name, algorithm in algorithms.items():
    cv_scores = cross_val_score(algorithm, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores
    }
# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Algorithm': list(results.keys()),
    'Mean CV Score': [results[name]['mean'] for name in results.keys()],
    'Std CV Score': [results[name]['std'] for name in results.keys()]
})
comparison_df = comparison_df.sort_values('Mean CV Score', ascending=False)
print("\n=== ALGORITHM COMPARISON ===")
print(comparison_df.to_string(index=False))
# Plot comparison
plt.figure(figsize=(12, 8))
plt.errorbar(comparison_df['Algorithm'], comparison_df['Mean CV Score'],
             yerr=comparison_df['Std CV Score'], fmt='o', capsize=5)
plt.xlabel('Algorithm')
plt.ylabel('Cross-validation Score')
plt.title('Algorithm Performance Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

# **5. Hyperparameter Tuning Best Practices**

```python
# Best practices checklist
best_practices = {
    "Data Preparation": [
        "Handle missing values appropriately",
        "Scale features when necessary",
        "Encode categorical variables",
        "Split data properly (train/validation/test)"
    ],
    "Model Selection": [
        "Start with simple models",
        "Use cross-validation for model selection
```

When to Use Each Algorithm:

**Classification:**

- **Logistic Regression**: Linear relationships, interpretability needed, baseline model
- **Decision Trees**: Non-linear relationships, interpretability crucial, mixed data types
- **Random Forest**: Robust performance needed, feature importance required
- **SVM**: High-dimensional data, clear margin separation, kernel flexibility needed
- **Neural Networks**: Complex patterns, large datasets, performance over interpretability

**Regression:**

- **Linear Regression**: Simple linear relationships, interpretability needed, baseline model
- **Polynomial Regression**: Non-linear relationships, known polynomial structure
- **Ridge Regression**: Multicollinearity present, all features relevant
- **Lasso Regression**: Feature selection needed, sparse models preferred
- **Decision Tree Regression**: Non-linear relationships, interpretability important

**Common Algorithms:**
*A. Clustering* 

- K-Means: Partitions data into K clusters.
- Hierarchical Clustering: Builds a tree of clusters.
- DBSCAN: Density-based clustering (good for irregular shapes).
- Gaussian Mixture Models (GMM): Probabilistic clustering.

*B. Dimensionality Reduction*
Reduces the number of features while preserving key information.

- **PCA (Principal Component Analysis)**: Linear projection to uncorrelated axes.
- **t-SNE**: Non-linear, preserves local structure (great for visualization).
- Autoencoders: Neural networks for compressed representations.
- Independent Component Analysis (ICA) is a machine learning technique used to separate a multivariate signal into its independent, non-Gaussian components, effectively identifying underlying factors or sources within mixed data.

# Complete Guide to Unsupervised Learning Algorithms

Unsupervised learning discovers hidden patterns in data without labeled examples. This comprehensive guide covers all major clustering and dimensionality reduction techniques with practical Python implementations.

## Table of Contents

1. [Clustering Algorithms](https://claude.ai/chat/922a6b2d-621f-47eb-8258-6f1f9d36e81c#clustering-algorithms)
2. [Dimensionality Reduction Algorithms](https://claude.ai/chat/922a6b2d-621f-47eb-8258-6f1f9d36e81c#dimensionality-reduction-algorithms)
3. [Algorithm Comparison and Selection](https://claude.ai/chat/922a6b2d-621f-47eb-8258-6f1f9d36e81c#algorithm-comparison-and-selection)
4. [Real-World Applications](https://claude.ai/chat/922a6b2d-621f-47eb-8258-6f1f9d36e81c#real-world-applications)

---

## Clustering Algorithms

Clustering groups similar data points together, revealing natural structures in unlabeled data.

### 1. K-Means Clustering

K-Means partitions data into K clusters by minimizing within-cluster sum of squares. It's simple, fast, and works well with spherical clusters.

**How it works:**

- Randomly initialize K cluster centers
- Assign each point to the nearest center
- Update centers to the mean of assigned points
- Repeat until convergence

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

# Generate sample data
np.random.seed(42)
X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# K-Means Implementation
print("=== K-MEANS CLUSTERING ===")

# Basic K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_blobs)

# Evaluate clustering
silhouette_kmeans = silhouette_score(X_blobs, kmeans_labels)
print(f"K-Means Silhouette Score: {silhouette_kmeans:.4f}")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original data
axes[0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='viridis', alpha=0.7)
axes[0].set_title('Original Data (True Clusters)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# K-Means results
axes[1].scatter(X_blobs[:, 0], X_blobs[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
axes[1].set_title('K-Means Clustering Results')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()

# Elbow method for optimal K
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_blobs)
    inertias.append(kmeans_temp.inertia_)

axes[2].plot(K_range, inertias, 'bo-')
axes[2].set_xlabel('Number of Clusters (K)')
axes[2].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[2].set_title('Elbow Method for Optimal K')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Advanced K-Means features
print("\nK-Means Advanced Features:")
print(f"Number of iterations: {kmeans.n_iter_}")
print(f"Inertia (WCSS): {kmeans.inertia_:.4f}")
print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")

```

### 2. Hierarchical Clustering

Hierarchical clustering builds a tree of clusters (dendrogram) by iteratively merging or splitting clusters.

**Two approaches:**

- **Agglomerative (Bottom-up):** Start with individual points, merge similar clusters
- **Divisive (Top-down):** Start with all points, split clusters recursively

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

print("\n=== HIERARCHICAL CLUSTERING ===")

# Agglomerative Clustering
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_blobs)

# Evaluate clustering
silhouette_hierarchical = silhouette_score(X_blobs, hierarchical_labels)
print(f"Hierarchical Clustering Silhouette Score: {silhouette_hierarchical:.4f}")

# Create dendrogram
plt.figure(figsize=(15, 8))

# Subplot 1: Clustering results
plt.subplot(2, 2, 1)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.7)
plt.title('Hierarchical Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Subplot 2: Dendrogram
plt.subplot(2, 2, 2)
linkage_matrix = linkage(X_blobs, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=12)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')

# Compare different linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']
linkage_scores = []

for method in linkage_methods:
    if method == 'ward':
        hierarchical_temp = AgglomerativeClustering(n_clusters=4, linkage=method)
    else:
        hierarchical_temp = AgglomerativeClustering(n_clusters=4, linkage=method)

    labels_temp = hierarchical_temp.fit_predict(X_blobs)
    score = silhouette_score(X_blobs, labels_temp)
    linkage_scores.append(score)

plt.subplot(2, 2, 3)
plt.bar(linkage_methods, linkage_scores)
plt.title('Linkage Methods Comparison')
plt.xlabel('Linkage Method')
plt.ylabel('Silhouette Score')
plt.xticks(rotation=45)

# Distance threshold clustering
plt.subplot(2, 2, 4)
hierarchical_distance = AgglomerativeClustering(n_clusters=None, distance_threshold=10, linkage='ward')
distance_labels = hierarchical_distance.fit_predict(X_blobs)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=distance_labels, cmap='viridis', alpha=0.7)
plt.title(f'Distance Threshold Clustering\n({hierarchical_distance.n_clusters_} clusters found)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

```

### 3. DBSCAN (Density-Based Spatial Clustering)

DBSCAN groups together points in high-density areas and marks outliers in low-density regions. It's excellent for irregular cluster shapes and noise detection.

**Key concepts:**

- **Core points:** Points with at least min_samples neighbors within eps distance
- **Border points:** Non-core points within eps distance of core points
- **Outliers:** Points that are neither core nor border points

```python
print("\n=== DBSCAN CLUSTERING ===")

# Generate complex-shaped data
X_complex = np.vstack([
    make_circles(n_samples=100, factor=0.3, noise=0.1, random_state=42)[0],
    make_moons(n_samples=100, noise=0.1, random_state=42)[0] + [2, 0],
    np.random.rand(20, 2) * 4 - 2  # Add some noise points
])

# DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_complex)

# Count clusters and outliers
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_outliers = list(dbscan_labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of outliers: {n_outliers}")

# Visualize DBSCAN results
plt.figure(figsize=(15, 10))

# Original complex data
plt.subplot(2, 3, 1)
plt.scatter(X_complex[:, 0], X_complex[:, 1], alpha=0.7)
plt.title('Complex-Shaped Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# DBSCAN results
plt.subplot(2, 3, 2)
unique_labels = set(dbscan_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Outliers in black
        col = 'black'
        marker = 'x'
        label = 'Outliers'
    else:
        marker = 'o'
        label = f'Cluster {k}'

    class_member_mask = (dbscan_labels == k)
    xy = X_complex[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, alpha=0.7, s=50)

plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Parameter sensitivity analysis
eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
min_samples_values = [3, 5, 7, 10]

param_results = []
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan_temp = DBSCAN(eps=eps, min_samples=min_samples)
        labels_temp = dbscan_temp.fit_predict(X_complex)
        n_clusters_temp = len(set(labels_temp)) - (1 if -1 in labels_temp else 0)
        n_outliers_temp = list(labels_temp).count(-1)
        param_results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters_temp,
            'n_outliers': n_outliers_temp
        })

# Visualize parameter effects
results_df = pd.DataFrame(param_results)
pivot_clusters = results_df.pivot(index='min_samples', columns='eps', values='n_clusters')
pivot_outliers = results_df.pivot(index='min_samples', columns='eps', values='n_outliers')

plt.subplot(2, 3, 4)
sns.heatmap(pivot_clusters, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Clusters vs Parameters')
plt.xlabel('eps')
plt.ylabel('min_samples')

plt.subplot(2, 3, 5)
sns.heatmap(pivot_outliers, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Outliers vs Parameters')
plt.xlabel('eps')
plt.ylabel('min_samples')

# Compare with K-Means on complex data
plt.subplot(2, 3, 6)
kmeans_complex = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_complex_labels = kmeans_complex.fit_predict(X_complex)
plt.scatter(X_complex[:, 0], X_complex[:, 1], c=kmeans_complex_labels, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_complex.cluster_centers_[:, 0], kmeans_complex.cluster_centers_[:, 1],
            c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means on Complex Data\n(Shows limitation)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

```

### 4. Gaussian Mixture Models (GMM)

GMM assumes data comes from a mixture of Gaussian distributions. It's a probabilistic clustering method that provides soft cluster assignments.

**Key features:**

- Soft clustering (probability of belonging to each cluster)
- Can handle elliptical cluster shapes
- Provides cluster covariance information

```python
print("\n=== GAUSSIAN MIXTURE MODELS ===")

# Generate elliptical clusters
np.random.seed(42)
X_elliptical = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100),
    np.random.multivariate_normal([3, 3], [[1, -0.5], [-0.5, 1]], 100),
    np.random.multivariate_normal([6, 0], [[0.5, 0], [0, 2]], 100)
])

# GMM clustering
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_elliptical)
gmm_probs = gmm.predict_proba(X_elliptical)

# Evaluate clustering
silhouette_gmm = silhouette_score(X_elliptical, gmm_labels)
print(f"GMM Silhouette Score: {silhouette_gmm:.4f}")
print(f"GMM Log-likelihood: {gmm.score(X_elliptical):.4f}")
print(f"GMM AIC: {gmm.aic(X_elliptical):.4f}")
print(f"GMM BIC: {gmm.bic(X_elliptical):.4f}")

# Visualize GMM results
plt.figure(figsize=(18, 12))

# Hard clustering
plt.subplot(2, 3, 1)
plt.scatter(X_elliptical[:, 0], X_elliptical[:, 1], c=gmm_labels, cmap='viridis', alpha=0.7)
plt.title('GMM Hard Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Soft clustering (probabilities)
plt.subplot(2, 3, 2)
plt.scatter(X_elliptical[:, 0], X_elliptical[:, 1], c=gmm_probs[:, 0], cmap='Reds', alpha=0.7)
plt.title('GMM Soft Clustering - Cluster 0 Probability')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()

plt.subplot(2, 3, 3)
plt.scatter(X_elliptical[:, 0], X_elliptical[:, 1], c=gmm_probs[:, 1], cmap='Greens', alpha=0.7)
plt.title('GMM Soft Clustering - Cluster 1 Probability')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()

# Model selection using AIC/BIC
n_components_range = range(1, 11)
aic_scores = []
bic_scores = []

for n_components in n_components_range:
    gmm_temp = GaussianMixture(n_components=n_components, random_state=42)
    gmm_temp.fit(X_elliptical)
    aic_scores.append(gmm_temp.aic(X_elliptical))
    bic_scores.append(gmm_temp.bic(X_elliptical))

plt.subplot(2, 3, 4)
plt.plot(n_components_range, aic_scores, 'bo-', label='AIC')
plt.plot(n_components_range, bic_scores, 'ro-', label='BIC')
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion')
plt.title('Model Selection: AIC vs BIC')
plt.legend()
plt.grid(True)

# Compare covariance types
covariance_types = ['full', 'tied', 'diag', 'spherical']
covariance_scores = []

for cov_type in covariance_types:
    gmm_temp = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42)
    gmm_temp.fit(X_elliptical)
    covariance_scores.append(gmm_temp.bic(X_elliptical))

plt.subplot(2, 3, 5)
plt.bar(covariance_types, covariance_scores)
plt.title('Covariance Types Comparison (BIC)')
plt.xlabel('Covariance Type')
plt.ylabel('BIC Score')
plt.xticks(rotation=45)

# Visualize Gaussian components
plt.subplot(2, 3, 6)
plt.scatter(X_elliptical[:, 0], X_elliptical[:, 1], c=gmm_labels, cmap='viridis', alpha=0.6)

# Plot ellipses representing Gaussian components
from matplotlib.patches import Ellipse
colors = ['red', 'blue', 'green']
for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
    v, w = np.linalg.eigh(covar)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = Ellipse(mean, v[0], v[1], 180 + angle, color=colors[i], alpha=0.3)
    plt.gca().add_patch(ell)

plt.title('GMM Gaussian Components')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

```

---

## Dimensionality Reduction Algorithms

Dimensionality reduction techniques compress high-dimensional data while preserving important information.

### 1. Principal Component Analysis (PCA)

PCA finds orthogonal axes (principal components) that capture maximum variance in the data.

**Key concepts:**

- Linear transformation
- Components are ranked by explained variance
- Reduces dimensionality while preserving most information

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

print("\n=== PRINCIPAL COMPONENT ANALYSIS (PCA) ===")

# Load high-dimensional data (digits dataset)
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"Original data shape: {X_digits.shape}")
print(f"Features: {X_digits.shape[1]} (8x8 pixel images)")

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_digits)

print(f"PCA reduced shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
print(f"Number of components needed: {pca.n_components_}")

# Visualize PCA results
plt.figure(figsize=(18, 12))

# Explained variance
plt.subplot(2, 4, 1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)

# 2D PCA visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_digits)

plt.subplot(2, 4, 2)
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_digits, cmap='tab10', alpha=0.6)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('2D PCA Visualization')
plt.colorbar(scatter)

# Original vs reconstructed images
pca_reconstruct = PCA(n_components=50)
X_pca_50 = pca_reconstruct.fit_transform(X_digits)
X_reconstructed = pca_reconstruct.inverse_transform(X_pca_50)

# Show some examples
for i in range(6):
    # Original image
    plt.subplot(2, 6, i + 7)
    plt.imshow(X_digits[i].reshape(8, 8), cmap='gray')
    plt.title(f'Original {y_digits[i]}')
    plt.axis('off')

    # Reconstructed image
    plt.subplot(2, 6, i + 13)
    plt.imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
    plt.title(f'Reconstructed {y_digits[i]}')
    plt.axis('off')

# Compare different numbers of components
components_range = [5, 10, 20, 50, 100]
reconstruction_errors = []

for n_comp in components_range:
    pca_temp = PCA(n_components=n_comp)
    X_temp = pca_temp.fit_transform(X_digits)
    X_temp_reconstructed = pca_temp.inverse_transform(X_temp)
    error = np.mean((X_digits - X_temp_reconstructed) ** 2)
    reconstruction_errors.append(error)

plt.subplot(2, 4, 3)
plt.plot(components_range, reconstruction_errors, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.title('PCA Components vs Reconstruction Error')
plt.grid(True)

plt.tight_layout()
plt.show()

```

### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is excellent for visualizing high-dimensional data by preserving local structure and revealing clusters.

**Key features:**

- Non-linear dimensionality reduction
- Preserves local neighborhoods
- Great for visualization but not for feature extraction

```python
from sklearn.manifold import TSNE

print("\n=== t-SNE (t-Distributed Stochastic Neighbor Embedding) ===")

# Apply t-SNE (using subset for speed)
np.random.seed(42)
subset_indices = np.random.choice(len(X_digits), 1000, replace=False)
X_subset = X_digits[subset_indices]
y_subset = y_digits[subset_indices]

# t-SNE with different parameters
tsne_params = [
    {'perplexity': 30, 'learning_rate': 200},
    {'perplexity': 5, 'learning_rate': 200},
    {'perplexity': 50, 'learning_rate': 200},
    {'perplexity': 30, 'learning_rate': 50}
]

plt.figure(figsize=(20, 15))

for i, params in enumerate(tsne_params):
    tsne = TSNE(n_components=2, random_state=42, **params)
    X_tsne = tsne.fit_transform(X_subset)

    plt.subplot(3, 3, i + 1)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
    plt.title(f"t-SNE (perplexity={params['perplexity']}, lr={params['learning_rate']})")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    if i == 0:
        plt.colorbar(scatter)

# Compare PCA vs t-SNE
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_subset)

tsne_default = TSNE(n_components=2, random_state=42)
X_tsne_default = tsne_default.fit_transform(X_subset)

plt.subplot(3, 3, 5)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
plt.title('PCA 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(3, 3, 6)
plt.scatter(X_tsne_default[:, 0], X_tsne_default[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
plt.title('t-SNE 2D')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# Show the effect of different distance metrics
print("t-SNE preserves local structure better than PCA for visualization")
print("PCA preserves global structure and is deterministic")
print("t-SNE is stochastic and focuses on local neighborhoods")

plt.tight_layout()
plt.show()

```

### 3. Autoencoders

Autoencoders are neural networks that learn compressed representations of data by reconstructing the input.

**Architecture:**

- Encoder: Compresses input to lower dimensions
- Decoder: Reconstructs original input from compressed representation
- Bottleneck: The compressed representation (latent space)

```python
# Note: This requires TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam

    print("\n=== AUTOENCODERS ===")

    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_digits_scaled = scaler.fit_transform(X_digits)

    # Define autoencoder architecture
    input_dim = X_digits_scaled.shape[1]  # 64 features
    encoding_dim = 32  # Compressed representation

    # Encoder
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)

    # Decoder
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # Autoencoder model
    autoencoder = Model(input_img, decoded)

    # Encoder model (for getting compressed representations)
    encoder = Model(input_img, encoded)

    # Decoder model
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # Compile and train
    autoencoder.compile(optimizer=Adam(0.001), loss='mse')

    # Train the autoencoder
    history = autoencoder.fit(X_digits_scaled, X_digits_scaled,
                             epochs=50, batch_size=32,
                             validation_split=0.2, verbose=0)

    # Get compressed representations
    X_encoded = encoder.predict(X_digits_scaled)
    X_reconstructed = autoencoder.predict(X_digits_scaled)

    # Visualize results
    plt.figure(figsize=(20, 12))

    # Training history
    plt.subplot(2, 5, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2D visualization of encoded representations
    # Use PCA to reduce 32D to 2D for visualization
    pca_encoded = PCA(n_components=2)
    X_encoded_2d = pca_encoded.fit_transform(X_encoded)

    plt.subplot(2, 5, 2)
    scatter = plt.scatter(X_encoded_2d[:, 0], X_encoded_2d[:, 1],
                         c=y_digits, cmap='tab10', alpha=0.6)
    plt.title('Autoencoder Latent Space (2D)')
    plt.xlabel('Encoded PC1')
    plt.ylabel('Encoded PC2')
    plt.colorbar(scatter)

    # Original vs reconstructed examples
    n_examples = 8
    for i in range(n_examples):
        # Original
        plt.subplot(4, n_examples, i + 1 + n_examples)
        plt.imshow(X_digits_scaled[i].reshape(8, 8), cmap='gray')
        plt.title(f'Original {y_digits[i]}')
        plt.axis('off')

        # Reconstructed
        plt.subplot(4, n_examples, i + 1 + 2*n_examples)
        plt.imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
        plt.title(f'Reconstructed {y_digits[i]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate reconstruction error
    mse = np.mean((X_digits_scaled - X_reconstructed) ** 2)
    print(f"Autoencoder reconstruction MSE: {mse:.6f}")

except ImportError:
    print("\n=== AUTOENCODERS ===")
    print("TensorFlow not available. Install with: pip install tensorflow")
    print("Autoencoders are neural networks that learn compressed representations")
    print("They consist of an encoder (compression) and decoder (reconstruction)")

```

### 4. Independent Component Analysis (ICA)

ICA separates mixed signals into independent components, useful for signal processing and feature extraction.

**Key concepts:**

- Assumes data is a linear mixture of independent sources
- Finds components that are statistically independent
- Useful for blind source separation

```python
from sklearn.decomposition import FastICA

print("\n=== INDEPENDENT COMPONENT ANALYSIS (ICA) ===")

# Generate mixed signals
np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# Original independent sources
s1 = np.sin(2 * time)  # Sine

```

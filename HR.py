# Import libraries
import pandas as pd
from lightgbm import early_stopping

# Load data
hr_data = pd.read_csv(r"C:\Users\tjf4x\Desktop\R projects\Python HR Attrition\WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(hr_data)

# Explore dataset
pd.set_option('display.max_columns', None)
hr_data.describe()

hr_data.info()

# Immediately dropping columns that are irrelevant or potentially sensitive.
hr_data.drop(['EmployeeNumber', 'Gender', 'MaritalStatus', 'Age'] , axis=1, inplace=True)

# Drop variables with standard deviation of 0
hr_data.drop(['EmployeeCount', 'StandardHours'] , axis=1, inplace=True)

# Checking for null values
hr_data.isnull().sum()

# Finding unique values in object columns
import matplotlib.pyplot as plt

# Select object columns
categorical_data = hr_data.select_dtypes(include='object')

# Loop over each object column and create a stacked bar chart
for column in categorical_data.columns:
    category_counts = categorical_data[column].value_counts()

    # Create a stacked bar chart
    category_counts.plot(kind='bar', stacked=True)
    plt.title(f'Stacked Bar Chart for {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

# Dropping Over18 category since all records are the same
hr_data.drop(['Over18'] , axis=1, inplace=True)

# Use factorize to convert objects into factors
for column in hr_data.select_dtypes(include='object').columns:
    hr_data[column] = pd.factorize(hr_data[column])[0]

# Investigating correlations between variables to see if any are redundant
import seaborn as sns

# Build a correlation matrix
correlation_matrix = hr_data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix for hr_data')
plt.show()

# Dropping multiple variables to reduce multicolinearity and BusinessTravel because it has very low correlation to attrition
hr_data.drop(['BusinessTravel','YearsAtCompany', 'TotalWorkingYears', 'MonthlyIncome', 'PerformanceRating', 'YearsWithCurrManager'], axis=1, inplace=True)
# There is very little correlation between hourly, daily, monthly, or yearly rate.

# Run baseline model
from sklearn.model_selection import train_test_split

# Set Attrition as target variable
X = hr_data.drop(columns=['Attrition'])
y = hr_data['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Verify row count
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Run a logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score


hr_log = LogisticRegression(max_iter=10000, random_state=42)

# Fit the model on the training data
hr_log.fit(X_train, y_train)

# Predict on the test set
y_pred_log_reg = hr_log.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_log_reg)
conf_matrix = confusion_matrix(y_test, y_pred_log_reg)
class_report = classification_report(y_test, y_pred_log_reg)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Running Random Forest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)

# Fit to training data
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)

# Print evaluation metrics
print(f"Random Forest Classifier Accuracy: {accuracy_rf:.2f}")
print("Confusion Matrix:")
print(conf_matrix_rf)
print("Classification Report:")
print(class_report_rf)

# Random Forest Regressor to investigate feature importance
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Get feature importances from the regressor
feature_importances = rf_regressor.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importance
print("Feature Importances:")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.show()

# Running Light Gradient Boosted Forest
import lightgbm as lgb

# Prepare the LightGBM dataset
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Set up the parameters for LightGBM
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,
    'random_state': 123
}

# Train the model
lgb_model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_test])

# Predict on the test set
y_pred_lgb = (lgb_model.predict(X_test) > 0.5).astype(int)

# Evaluate the LightGBM model
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
conf_matrix_lgb = confusion_matrix(y_test, y_pred_lgb)
class_report_lgb = classification_report(y_test, y_pred_lgb)

# Print evaluation metrics
print(f"LightGBM Model Accuracy: {accuracy_lgb:.2f}")
print("Confusion Matrix:")
print(conf_matrix_lgb)
print("Classification Report:")
print(class_report_lgb)

# Campre recall of each model
recall_log_reg = recall_score(y_test, y_pred_log_reg)
recall_rf = recall_score(y_test, y_pred_rf)
recall_lgb = recall_score(y_test, y_pred_lgb)


print(f"Logistic Regression Recall: {recall_log_reg:.2f}")
print(f"Random Forest Recall: {recall_rf:.2f}")
print(f"LightGBM Recall: {recall_lgb:.2f}")

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "LightGBM"],
    "Accuracy": [accuracy, accuracy_rf, accuracy_lgb]
})
print(comparison_df)

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


# Initial EDA
def quick_eda(df):
    print("\n=== Data Overview ===")
    print(df.info())

    print("\n=== Missing Values ===")
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    print(missing)

    print("\n=== Numeric Features ===")
    print(df.describe())

    print("\n=== Categorical Features ===")
    print(df.describe(include=['O']))


quick_eda(train_df)

# Visualizations
plt.figure(figsize=(12, 8))
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('Sale Price Distribution')
plt.show()

# Correlation analysis
corr_matrix = train_df.corr(numeric_only=True)
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Top correlated features with SalePrice
top_corr = corr_matrix['SalePrice'].sort_values(ascending=False)[1:11]
print("\nTop 10 features correlated with SalePrice:")
print(top_corr)


# Feature Engineering
def prepare_data(df):
    # Handle missing values
    # Fill numerical missing values with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical missing values with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Feature transformations
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['Age'] = df['YrSold'] - df['YearBuilt']

    # Convert categorical to dummy variables
    df = pd.get_dummies(df)

    return df


# Prepare data
train_processed = prepare_data(train_df)

# Select features based on correlation
selected_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
    'FullBath', 'YearBuilt', 'TotalSF', 'TotalBath', 'Age',
    'TotRmsAbvGrd', 'Fireplaces', 'GarageArea'
]

# Add dummy columns that might have been created
features = [f for f in selected_features if f in train_processed.columns] + \
           [col for col in train_processed.columns if 'Neighborhood_' in col]

X = train_processed[features]
y = train_processed['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10),
    'Lasso Regression': Lasso(alpha=0.1)
}

results = {}
for name, model in models.items():
    # Create pipeline with standardization
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = pipeline.score(X_test, y_test)

    results[name] = {
        'RMSE': rmse,
        'R2 Score': r2,
        'Model': pipeline
    }

    print(f"\n{name} Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

# Feature importance for the best model
best_model = results['Ridge Regression']['Model']
coefficients = best_model.named_steps['ridge'].coef_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': coefficients
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Important Features')
plt.show()

# Make predictions on test set (for Kaggle submission)
test_processed = prepare_data(test_df)
X_kaggle_test = test_processed[features]

# Ensure all columns are present (some neighborhoods might be missing)
missing_cols = set(X.columns) - set(X_kaggle_test.columns)
for col in missing_cols:
    X_kaggle_test[col] = 0
X_kaggle_test = X_kaggle_test[X.columns]  # Ensure same column order

final_predictions = best_model.predict(X_kaggle_test)

# Create submission file
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': final_predictions
})
submission.to_csv('data/submission.csv', index=False)
print("\nSubmission file created!")
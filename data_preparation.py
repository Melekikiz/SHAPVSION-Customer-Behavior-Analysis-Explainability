import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('SwissData_Customer_Simulated.csv')
print("✅ Data loaded successfully")
print("Shape:", df.shape)
print("\nInitial missing values:\n", df.isnull().sum())

# Missing Values 
df['InvestmentAmountLast12M'].fillna(df['InvestmentAmountLast12M'].mean(), inplace=True)
df['RiskTolerance'].fillna('Medium', inplace=True)

# Data Corrections 
df['InvestmentAmountLast12M'] = df['InvestmentAmountLast12M'].clip(lower=0)
df['AnnualIncome'] = df['AnnualIncome'].clip(lower=20000)

#  Feature Engineering 
# Avoid division by zero
df['NumberOfInvestmentsLast12M'] = df['NumberOfInvestmentsLast12M'].replace(0, np.nan)
df['AvgInvestmentSize'] = df['InvestmentAmountLast12M'] / df['NumberOfInvestmentsLast12M']
df['AvgInvestmentSize'].fillna(0, inplace=True)  # restore zeros for clarity

df['InvestmentActivityScore'] = df['InteractionScore'] * df['InvestmentAmountLast12M']

# Map risk levels to numeric
risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df['RiskScore'] = df['RiskTolerance'].map(risk_mapping)

#Quality Check 
print("\nPost-cleaning summary:")
print(df.describe(include='all').T.head(10))
print("\nRemaining missing values:\n", df.isna().sum())

# Save cleaned dataset
df.to_csv('SwissData_Customers_Cleaned.csv', index=False)
df.to_parquet('SwissData_Customers_Cleaned.parquet', index=False)

print("\n✅ Cleaned data saved successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

import pandas as pd
import numpy as np
import datetime as dt

np.random.seed(42)
num_customers=1000
locations=['Zurich', 'Basel', 'Geneva', 'Bern', 'Lausanne']
genders=['Male', 'Female']
risk_levels=['Low', 'Medium', 'High']

#Customer Demographics
customer_ids=range(1,num_customers + 1)
names = [f"Customer_{i}" for i in customer_ids]
ages = np.random.randint(25, 70, size=num_customers)
genders_list=np.random.choice(genders, size=num_customers, p=[0.52, 0.48])
locations_list = np.random.choice(locations, size=num_customers)

#Financial Information
annual_income = np.random.normal(120000, 40000, size=num_customers)
annual_income = np.where(annual_income < 20000, 20000, annual_income)
investment_amount = np.random.normal(annual_income*0.15, annual_income*0.05)
investment_amount = np.where(investment_amount < 0, 0, investment_amount)
num_investments=np.random.poisson(5, size=num_customers)

#Risk an Behavior Score
risk_tolerance = np.random.choice(risk_levels, size=num_customers, p=[0.4, 0.4, 0.2])
interaction_score = np.random.randint(1, 101, size=num_customers)
customer_since = np.random.randint(1, 15, size=num_customers)
satisfaction_score = np.random.randint(1, 11, size=num_customers)
churned = np.random.choice([0, 1], size=num_customers, p=[0.8, 0.2])
signup_dates = [dt.date(2025, 1, 1) - dt.timedelta(days=np.random.randint(0, 5 * 365)) for _ in range(num_customers)]

#Add missing values 
for _ in range(int(num_customers*0.05)):
    idx = np.random.randint(0, num_customers)
    investment_amount[idx] = np.nan

for _ in range(int(num_customers*0.03)):
    idx=np.random.randint(0, num_customers)
    risk_tolerance[idx] = None

for _ in range(int(num_customers*0.02)):
    idx = np.random.randint(0, num_customers)
    investment_amount[idx] = -abs(np.random.normal(50000, 20000))

#Build DataFrame
df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Name': names,
    'Age': ages,
    'Gender': genders_list,
    'Location': locations_list,
    'AnnualIncome': annual_income,
    'InvestmentAmountLast12M': investment_amount,
    'NumberOfInvestmentsLast12M': num_investments,
    'RiskTolerance': risk_tolerance,
    'InteractionScore': interaction_score,
    'CustomerTenure': customer_since,
    'SatisfactionScore': satisfaction_score,
    'Churned': churned,
    'SignupDate': signup_dates
})

# Clean-up and save
df['InvestmentAmountLast12M'] = df['InvestmentAmountLast12M'].clip(lower=0)
df.to_csv('SwissData_Customer_Simulated.csv', index=False)
df.to_parquet('SwissData_Customer_Simulated.parquet', index=False)

print("✅ Dataset created successfully!")
print("Shape:", df.shape)
print(df.head())

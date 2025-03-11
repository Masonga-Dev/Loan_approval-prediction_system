import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 500

# Generate synthetic data
np.random.seed(42)
data = {
    "Married": np.random.choice(["No", "Yes"], num_samples),
    "Dependents": np.random.choice(["0", "1", "2", "3+"], num_samples),
    "Education": np.random.choice(["Not Graduate", "Graduate"], num_samples),
    "Employment_Type": np.random.choice(["No", "Yes"], num_samples),
    "Income": np.random.randint(100000, 5000000, num_samples),
    "CoapplicantIncome": np.random.randint(0, 2000000, num_samples).astype(float),
    "LoanAmount": np.random.randint(50000, 3000000, num_samples).astype(float),
    "Loan_Amount_Term": np.random.randint(6, 60, num_samples).astype(float),
    "Credit_History": np.random.choice([0.0, 1.0], num_samples),
    "Property_Area": np.random.choice(["Rural", "Semiurban", "Urban"], num_samples),
    "Loan_Status": np.random.choice(["N", "Y"], num_samples)
}

# Adjust LoanAmount according to Income and typical loan rules
for i in range(num_samples):
    max_loan_amount = (data["Income"][i] + data["CoapplicantIncome"][i]) * 0.6
    data["LoanAmount"][i] = min(data["LoanAmount"][i], max_loan_amount)

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("loan_data.csv", index=False)

print("loan_data.csv file has been generated.")
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv("loan_data.csv")

# Columns to encode (convert text to numbers)
categorical_columns = ['Married', 'Education', 'Employment_Type', 'Property_Area', 'Loan_Status']

# Apply Label Encoding
encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# Convert 'Dependents' column to numeric
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Features (X) - Remove 'Loan_Status' because it's what we are predicting
X = df.drop(columns=['Loan_Status'])

# Target (y) - The column we want to predict
y = df['Loan_Status']

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale the features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create the AI model with increased max_iter
model = LogisticRegression(max_iter=200)

# Train the model using training data
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Save the trained model and scaler
joblib.dump(model, "trained_model.pkl")
joblib.dump(scaler, "scaler.pkl")
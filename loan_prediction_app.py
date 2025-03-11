import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("trained_model.pkl")

# Load the scaler
scaler = joblib.load("scaler.pkl")

# Function to take user input and predict loan approval
def predict_loan():
    st.markdown("<h1 style='text-align: center;'>Loan Approval System</h1>", unsafe_allow_html=True)
    st.write("Please provide the following details to check your loan eligibility:")

    # Get user input
    married = st.selectbox("Married?", ("No", "Yes"))
    dependents = st.selectbox("Number of Dependents", ("0", "1", "2", "3+"))
    education = st.selectbox("Education Level", ("Not Graduate", "Graduate"))
    employment_type = st.selectbox("Self-Employed?", ("No", "Yes"))
    income = st.number_input("Applicant Income (in RWF)", min_value=100000, max_value=5000000, step=10000)
    coapplicant_income = st.number_input("Co-applicant Income (in RWF)", min_value=0, max_value=2000000, step=10000)
    loan_amount = st.number_input("Loan Amount Requested (in RWF)", min_value=50000, max_value=3000000, step=10000)
    loan_term = st.number_input("Loan Term (in months, e.g., 6 to 60)", min_value=6, max_value=60, step=1)
    credit_history = st.selectbox("Credit History", ("Bad", "Good"))
    property_area = st.selectbox("Property Area", ("Rural", "Semiurban", "Urban"))

    if st.button("Submit"):
        # Convert categorical inputs to numeric
        married = 1 if married == "Yes" else 0
        dependents = 3 if dependents == "3+" else int(dependents)
        education = 1 if education == "Graduate" else 0
        employment_type = 1 if employment_type == "Yes" else 0
        credit_history = 1 if credit_history == "Good" else 0
        property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
        
        # Create a dataframe for prediction
        user_data = [[married, dependents, education, employment_type, income, coapplicant_income,
                      loan_amount, loan_term, credit_history, property_area]]
        
        # Scale the user input data
        user_data = scaler.transform(user_data)
        
        # Make prediction
        prediction = model.predict(user_data)

        # Show result
        if prediction[0] == 1:
            st.success("✅ Loan Approved! 🎉")
        else:
            st.error("❌ Loan Rejected! 😞")

# Function for the Home page
def home():
    st.markdown("<h1 style='text-align: center;'>Welcome to the Loan Approval System</h1>", unsafe_allow_html=True)
    st.write("Use the navigation menu to explore the app.")
    
    # Explanation card
    st.markdown("""
    <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
        <h3>About the Loan Approval System</h3>
        <p>This AI-powered system helps you determine your loan eligibility based on your financial data. 
        Simply provide the required details, and our AI will predict your loan approval chances instantly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Image card with button
    st.markdown("""
    <div style="background-color: black; padding: 20px; border-radius: 10px; text-align: center;">
    """, unsafe_allow_html=True)
    st.image("loan.jpg", use_container_width=True)
    st.markdown("""
        <br><br>
    """, unsafe_allow_html=True)
    
    if st.button("Check Your Loan Eligibility"):
        st.session_state.page = "Loan Approval"

# Function for the About Us page
def about_us():
    st.markdown("<h1 style='text-align: center;'>About Us</h1>", unsafe_allow_html=True)
    st.write("""
        🚀 **AI-Powered Loan Approval & Prediction System**
        
        🔹 **Instant Decisions** – Get quick insights on loan approval based on financial data.
        
        🔹 **Smart & Fair Analysis** – AI evaluates income, credit history, and risk factors without bias.
        
        🔹 **Efficient for Banks & Lenders** – Helps financial institutions make data-driven loan approvals.
        
        🔹 **User-Friendly Process** – Simply enter details, and our AI predicts approval chances instantly.
        
        💡 **Transforming Lending with Speed, Accuracy & Transparency!** 🚀💰
    """)

# Function for the Contact Us page
def contact_us():
    st.markdown("<h1 style='text-align: center;'>Contact Us</h1>", unsafe_allow_html=True)
    st.write("""
        📞 **Telephone:** +250786781268
        
        📩 **Email:** loanapprovalhelp@gmail.com
    """)

# Navigation
st.sidebar.title("Navigation")
if "page" not in st.session_state:
    st.session_state.page = "Home"

page = st.sidebar.radio("Go to", ["Home", "Loan Approval", "About Us", "Contact Us"], index=["Home", "Loan Approval", "About Us", "Contact Us"].index(st.session_state.page))

if page == "Home":
    home()
elif page == "Loan Approval":
    predict_loan()
elif page == "About Us":
    about_us()
elif page == "Contact Us":
    contact_us()

# Add custom CSS for styling
st.markdown("""
    <style>
        .btn-primary {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
        }
        .btn-primary:hover {
            background-color: #0056b3;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stTextInput>div>div>input {
            border: 1px solid #007bff;
            border-radius: 5px;
        }
        .stSelectbox>div>div>div>div {
            border: 1px solid #007bff;
            border-radius: 5px;
        }
        .footer {
            background-color: black;
            color: white;
            padding: 10px;
            text-align: center;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Add footer
st.markdown("""
    <div class="footer">
        <p>&copy; 2025 Loan Approval System. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import re

# Function to clean and transform the data
def clean_and_transform_dataframe(df):
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    df['To_Account_id'] = df['To_Account_id'].astype(int)
    df['From_Account_id'] = df['From_Account_id'].apply(lambda x: re.sub(r'[^\x20-\x7E]', '', str(x)))
    df['From_Account_id'] = df['From_Account_id'].apply(lambda x: re.sub(r'(\.\d+)\.', '\1', str(x)))
    df['From_Account_id'] = pd.to_numeric(df['From_Account_id'], errors='coerce')
    df.dropna(subset=['From_Account_id'], inplace=True)
    df['From_Account_id'] = df['From_Account_id'].astype(int)
    df['amount'] = df['amount'].apply(lambda x: re.sub(r'[^\x20-\x7E]', '', str(x)))
    df['amount'] = df['amount'].apply(lambda x: re.sub(r'(\.\d+)\.', '\1', str(x)))
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df.dropna(subset=['amount'], inplace=True)
    df['amount'] = df['amount'].astype(int)
    df['frequency'] = df.groupby('To_Account_id')['To_Account_id'].transform('count')
    return df

# Function to train the Isolation Forest model
def train_isolation_forest(df):
    features = ['amount', 'To_Account_id', 'From_Account_id', 'frequency']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    df['anomaly_score'] = model.fit_predict(X_scaled)
    df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
    return model, df

# Streamlit App
st.title("Anomaly Detection in Transactions")
st.write("Upload a dataset to detect anomalies using an Isolation Forest model.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Load the data
    st.write("### Dataset Preview")
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Clean and transform the data
    st.write("### Cleaning and Transforming the Data")
    df_cleaned = clean_and_transform_dataframe(df)
    st.write("Data after cleaning and transformation:")
    st.write(df_cleaned.head())

    # Train the Isolation Forest model
    st.write("### Training the Isolation Forest Model")
    model, df_with_anomalies = train_isolation_forest(df_cleaned)

    # Display results
    st.write("### Results")
    st.write("Number of anomalies detected:")
    num_anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1].shape[0]
    st.write(f"**{num_anomalies} anomalies detected**")

    st.write("Anomalies (Sample):")
    anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1]
    st.write(anomalies[['To_Account_id', 'From_Account_id', 'amount', 'anomaly_score']].head())

    # Allow user to download the result
    st.write("### Download Results")
    csv = df_with_anomalies.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="anomaly_detection_results.csv",
        mime="text/csv"
    )

    # Optional: Show feature importance (if applicable)
    st.write("### Feature Summary")
    st.write("Feature statistics after cleaning:")
    st.write(df_cleaned.describe())

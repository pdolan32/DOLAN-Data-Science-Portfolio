import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes, load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Title of the app
st.title('Machine Learning Dataset Analysis')

# Sidebar for user input
option = st.sidebar.selectbox('Choose Dataset Option', ('Upload Your Own', 'Use Sample Dataset'))

# Upload Dataset
if option == 'Upload Your Own':
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type='csv')
    
    if uploaded_file is not None:
        # Read the uploaded file into a pandas dataframe
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.write(df)

# Use Sample Dataset
else:
    # Sidebar for dataset selection
    dataset_option = st.sidebar.selectbox('Choose Sample Dataset', ('California Housing', 'Breast Cancer', 'Diabetes', 'Iris'))

    if dataset_option == 'California Housing':
        # Load California Housing dataset
        housing = fetch_california_housing()
        df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
        df['MedHouseValue'] = housing.target
        st.write("Sample Dataset: California Housing")
        st.write(df.head())

    elif dataset_option == 'Breast Cancer':
        # Load Breast Cancer dataset
        cancer = load_breast_cancer()
        df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
        df['malignant'] = cancer.target
        st.write("Sample Dataset: Breast Cancer")
        st.write(df.head())

    elif dataset_option == 'Diabetes':
        diabetes = load_diabetes()
        df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
        df['disease_progression'] = diabetes.target
        st.write("Sample Dataset: Diabetes")
        st.write(df.head())

    elif dataset_option == 'Iris':
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        st.write("Sample Dataset: Iris")
        st.write(df)

model_option = st.sidebar.selectbox('Choose a Supervised Machine Learning Model', ('None', 'Linear Regression', 'Logistic Regression'))

if model_option == 'Linear Regression':
    # Select Features and Target
    st.sidebar.subheader('Select Features and Target for Linear Regression')
    test_size = st.sidebar.slider('Test Size for Train-Test Split', min_value=0.1, max_value=0.9, step=0.05, value=0.2)
    target = st.sidebar.selectbox('Choose Target Variable', df.columns)
    available_features = [col for col in df.columns if col != target]
    features = st.sidebar.multiselect('Choose Features', available_features)
    
    if target and features:
        # Prepare data for model
        X = df[features]
        y = df[target]
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert scaled data back into DataFrame and retain original feature names
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        # Split the scaled data
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        # Initialize and train the linear regression model on scaled data
        lin_reg_scaled = LinearRegression()
        lin_reg_scaled.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred_scaled = lin_reg_scaled.predict(X_test_scaled)
        
        # Evaluate model performance
        mse_scaled = mean_squared_error(y_test, y_pred_scaled)
        rmse_scaled = root_mean_squared_error(y_test, y_pred_scaled)
        r2_scaled = r2_score(y_test, y_pred_scaled)

        # Display the evaluation metrics in Streamlit
        st.write("Scaled Data Model Evaluation Metrics:")
        st.write(f'Mean Squared Error: {mse_scaled:.2f}')
        st.write(f'Root Mean Squared Error: {rmse_scaled:.2f}')
        st.write(f'R² Score: {r2_scaled:.2f}')

        st.write("\nModel Coefficients (Scaled):")
        st.write(pd.Series(lin_reg_scaled.coef_, index=features))
        st.write("\nModel Intercept (Scaled):")
        st.write(lin_reg_scaled.intercept_)

if model_option == 'Logistic Regression':

    # Select Features and Target
    st.sidebar.subheader('Select Features and Target for Logistic Regression')
    test_size = st.sidebar.slider('Test Size for Train-Test Split', min_value=0.1, max_value=0.9, step=0.05, value=0.2)
    target = st.sidebar.selectbox('Choose Target Variable', df.columns)
    available_features = [col for col in df.columns if col != target]
    features = st.sidebar.multiselect('Choose Features', available_features)
    
    if target and features:
        # Prepare data for model
        X = df[features]
        y = df[target]

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Convert scaled data back into DataFrame and retain original feature names
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        # Split the scaled data
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        # Initialize and train the logistic regression model
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        # Make predictions on the test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # Display classification report
        report = classification_report(y_test, y_pred)
        st.markdown("### Classification Report")
        st.markdown(f'```\n{report}\n```')

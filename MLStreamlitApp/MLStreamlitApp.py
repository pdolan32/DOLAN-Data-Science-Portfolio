import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes, load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Title of the app
st.title('Machine Learning Dataset Analysis')

st.write('To begin, either upload a dataset or choose a sample dataset from the dropdown menu.')

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
        st.write('This is the California Housing dataset.' \
        ' The California Housing Dataset is a popular dataset often used for regression tasks in machine learning.' \
        ' It contains data collected from the 1990 California census and is used to predict median house value in different districts based on various features.')
        housing = fetch_california_housing()
        df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
        df['MedHouseValue'] = housing.target
        st.header("Sample Dataset: California Housing")
        st.write(df.head())
        st.write('To analyze this dataset, please choose a supervised machine learning model from the sidebar.' \
        ' This dataset works best with regression models (like Linear Regression).')

    elif dataset_option == 'Breast Cancer':
        # Load Breast Cancer dataset
        st.write('This is the Breast Cancer dataset.' \
        ' This dataset is a well-known dataset used for binary classification, often to test models that distinguish between benign and malignant tumors: The target value is the diagnosis, with 0 = malignant and 1 = benign.')
        cancer = load_breast_cancer()
        df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
        df['diagnosis'] = cancer.target
        st.header("Sample Dataset: Breast Cancer")
        st.write(df.head())
        st.write('To analyze this dataset, please choose a supervised machine learning model from the sidebar.' \
        ' This dataset works best with classification models (like Logistic Regression and Decision Trees).')

    elif dataset_option == 'Diabetes':
        st.write('This is the Diabetes dataset.' \
        ' The Diabetes dataset is another classic dataset, mostly used for regression tasks. It comes from a medical study and is commonly used to predict disease progression based on health indicators. ' \
        ' The goal is to predict a quantitative measure of disease progression one year after baseline (e.g., how much worse or better their diabetes got).')
        diabetes = load_diabetes()
        df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
        df['disease_progression'] = diabetes.target
        st.header("Sample Dataset: Diabetes")
        st.write(df.head())
        st.write('To analyze this dataset, please choose a supervised machine learning model from the sidebar.' \
        ' This dataset works best with regression models (like Linear Regression).')

    elif dataset_option == 'Iris':
        st.write('This is the Iris dataset.' \
        ' The Iris dataset contains information about 150 iris flowers from 3 different species.' \
        ' Each sample represents one flower, and the goal is to classify the species of the flower based on 4 features.' \
        ' The species of the Iris flower is categorical (0, 1, 2): Iris setosa, Iris versicolor, Iris virginica, respectively.')
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        st.header("Sample Dataset: Iris")
        st.write(df.head())
        st.write('To analyze this dataset, please choose a supervised machine learning model from the sidebar.' \
        ' This dataset works best with classification models (like Logistic Regression and Decision Trees).')

model_option = st.sidebar.selectbox('Choose a Supervised Machine Learning Model', ('None', 'Linear Regression', 'Logistic Regression', 'Decision Tree'))

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
        st.header("Scaled Data Model Evaluation Metrics:")
        st.write(f'Mean Squared Error: {mse_scaled:.2f}')
        st.write(f'Root Mean Squared Error: {rmse_scaled:.2f}')
        st.write(f'R² Score: {r2_scaled:.2f}')

        st.header("\nModel Coefficients (Scaled):")
        st.write(pd.Series(lin_reg_scaled.coef_, index=features))
        st.header("\nModel Intercept (Scaled):")
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

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.header("Confusion Matrix")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.pyplot(fig)

        # Display classification report
        report = classification_report(y_test, y_pred)
        st.header("Classification Report")
        st.markdown(f'```\n{report}\n```')

        # Check number of unique classes in target
        unique_classes = y_test.nunique()

        if unique_classes == 2:
            # Binary classification → show ROC curve
            try:
                y_probs = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)

                # Plot ROC Curve
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax_roc.legend(loc="lower right")
                st.header("ROC Curve")
                st.write(f"ROC AUC Score: {roc_auc:.2f}")
                st.pyplot(fig_roc)

            except Exception as e:
                st.error(f"Error computing ROC curve: {e}")
        else:
            # Multiclass → show message
            st.write("ROC curve is only available for binary classification problems.")

if model_option == 'Decision Tree':

    # Select Features and Target
    st.sidebar.subheader('Select Features and Target for Decision Tree')
    test_size = st.sidebar.slider('Test Size for Train-Test Split', min_value=0.1, max_value=0.9, step=0.05, value=0.2)
    target = st.sidebar.selectbox('Choose Target Variable', df.columns)
    available_features = [col for col in df.columns if col != target]
    features = st.sidebar.multiselect('Choose Features', available_features)

    # NEW: Tree hyperparameter sliders
    max_depth = st.sidebar.slider('Max Depth of Decision Tree', min_value=1, max_value=20, value=3)
    min_samples_split = st.sidebar.slider('Min Samples Split', min_value=2, max_value=20, value=2)
    min_samples_leaf = st.sidebar.slider('Min Samples Leaf', min_value=1, max_value=20, value=1)

    if target and features:
        # Prepare data for model
        X = df[features]
        y = df[target]

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        # NEW: Initialize with custom hyperparameters
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        st.header("Confusion Matrix:")
        st.write(f"Accuracy: {accuracy:.2f}")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # Display classification report
        report = classification_report(y_test, y_pred)
        st.header("Classification Report")
        st.markdown(f'```\n{report}\n```')

        # Dynamic class names
        class_names = [str(cls) for cls in np.unique(y_train)]

        # Graphviz chart
        dot_data = tree.export_graphviz(
            model,
            feature_names=X_train_scaled.columns,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        st.header("Decision Tree")
        st.graphviz_chart(dot_data)

       # Check number of unique classes in target
        unique_classes = y_test.nunique()

        if unique_classes == 2:
            # Binary classification → show ROC curve
            try:
                y_probs = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)

                # Plot ROC Curve
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax_roc.legend(loc="lower right")
                st.header("ROC Curve")
                st.write(f"ROC AUC Score: {roc_auc:.2f}")
                st.pyplot(fig_roc)

            except Exception as e:
                st.error(f"Error computing ROC curve: {e}")
        else:
            # Multiclass → show message
            st.write("ROC curve is only available for binary classification problems.")


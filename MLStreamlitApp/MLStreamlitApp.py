# Import the required libraries
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

# Write in the title of the app
st.title('Supervised Machine Learning Dataset Analysis')

# Write in the preliminary app instructions
st.write('To begin, consult the sidebar to either upload a dataset or choose from a sample of datasets.')

# Creates a sidebar for user input
# Using the radio widget, the user is presented with the option to either upload a dataset or choose a sample dataset
option = st.sidebar.radio('Choose Dataset Option', ('Upload Your Own', 'Use Sample Dataset'))

if option == 'Upload Your Own': # If the user chooses to upload their own dataset
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type='csv') # A variable for the user's uploaded dataset
    
    if uploaded_file is not None:
        # Read the uploaded dataset file into a pandas dataframe
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.write(df.head()) # The first five rows of the dataset are displayed
        st.subheader("Summary Statistics:")
        st.write(df.describe()) # Summary statistics and information are displayed

else: # If the user chooses to select a sample dataset
    # A selectbox is created in the sidebar for the user to choose from a variety of sample datasets
    dataset_option = st.sidebar.selectbox('Choose Sample Dataset', ('California Housing', 'Breast Cancer', 'Diabetes', 'Iris'))

    if dataset_option == 'California Housing': # If the user chooses the California Housing Dataset (the default sample option)
        # Description of the dataset
        st.write('This is the California Housing dataset.' \
        ' The California Housing Dataset is a popular dataset often used for regression tasks in machine learning.' \
        ' It contains data collected from the 1990 California census and is used to predict median house value in different districts based on various features.')
        housing = fetch_california_housing() # load in the dataset from sklearn.datasets
        df = pd.DataFrame(data=housing.data, columns=housing.feature_names) # load in the dataset as a pandas dataframe
        df['MedHouseValue'] = housing.target # display the target variable in the dataframe
        st.header("Sample Dataset: California Housing")
        st.write(df.head()) # The first five rows of the dataset are displayed
        st.subheader("Summary Statistics:")
        st.write(df.describe()) # Summary statistics and information are displayed
        # Further instructions are suggested
        st.write('To analyze this dataset, please choose a supervised machine learning model from the sidebar.' \
        ' This dataset works best with regression models (like Linear Regression).')

    elif dataset_option == 'Breast Cancer': # If the user chooses the Breast Cancer Dataset
        # Description of the dataset
        st.write('This is the Breast Cancer dataset.' \
        ' This dataset is a well-known dataset used for binary classification, often to test models that distinguish between benign and malignant tumors: The target value is the diagnosis, with 0 = malignant and 1 = benign.')
        cancer = load_breast_cancer() # load in the dataset from sklearn.datasets
        df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names) # load in the dataset as a pandas dataframe
        df['diagnosis'] = cancer.target # display the target variable in the dataframe
        st.header("Sample Dataset: Breast Cancer")
        st.write(df.head()) # The first five rows of the dataset are displayed
        st.subheader("Summary Statistics:")
        st.write(df.describe()) # Summary statistics and information are displayed
        # Further instructions are suggested
        st.write('To analyze this dataset, please choose a supervised machine learning model from the sidebar.' \
        ' This dataset works best with classification models (like Logistic Regression and Decision Trees).')

    elif dataset_option == 'Diabetes': # If the user chooses the Diabetes Dataset
        # Description of the dataset
        st.write('This is the Diabetes dataset.' \
        ' The Diabetes dataset is another classic dataset, mostly used for regression tasks. It comes from a medical study and is commonly used to predict disease progression based on health indicators. ' \
        ' The goal is to predict a quantitative measure of disease progression one year after baseline (e.g., how much worse or better their diabetes got).')
        diabetes = load_diabetes() # load in the dataset from sklearn.datasets
        df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)  # load in the dataset as a pandas dataframe
        df['disease_progression'] = diabetes.target # display the target variable in the dataframe
        st.header("Sample Dataset: Diabetes") 
        st.write(df.head()) # The first five rows of the dataset are displayed
        st.subheader("Summary Statistics:")
        st.write(df.describe()) # Summary statistics and information are displayed
        # Further instructions are suggested
        st.write('To analyze this dataset, please choose a supervised machine learning model from the sidebar.' \
        ' This dataset works best with regression models (like Linear Regression).')

    elif dataset_option == 'Iris': # If the user chooses the Iris Dataset
        # Description of the dataset
        st.write('This is the Iris dataset.' \
        ' The Iris dataset contains information about 150 iris flowers from 3 different species.' \
        ' Each sample represents one flower, and the goal is to classify the species of the flower based on 4 features.' \
        ' The species of the Iris flower is categorical (0, 1, 2): Iris setosa, Iris versicolor, Iris virginica, respectively.')
        iris = load_iris() # load in the dataset from sklearn.datasets
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names) # load in the dataset as a pandas dataframe
        df['species'] = iris.target # display the target variable in the dataframe
        st.header("Sample Dataset: Iris")
        st.write(df.head()) # The first five rows of the dataset are displayed
        st.subheader("Summary Statistics:")
        st.write(df.describe()) # Summary statistics and information are displayed
        # Further instructions are suggested
        st.write('To analyze this dataset, please choose a supervised machine learning model from the sidebar.' \
        ' This dataset works best with classification models (like Logistic Regression and Decision Trees).')

# In the sidebar, the user is also presented with a selectbox to choose the supervised learning model they would like to apply to the dataset
model_option = st.sidebar.selectbox('Choose a Supervised Machine Learning Model', ('None', 'Linear Regression', 'Logistic Regression', 'Decision Tree'))

if model_option == 'Linear Regression': # If the user chooses Linear Regression
    st.sidebar.subheader('Select Features and Target for Linear Regression')
    # Using a slider, the user has the option to set and adjust the test size for the train-test split
    test_size = st.sidebar.slider('Test Size for Train-Test Split', min_value=0.1, max_value=0.9, step=0.05, value=0.2)
    # Using a selectbox, the user has the option to choose the target and feature variables in the analysis
    target = st.sidebar.selectbox('Choose Target Variable', df.columns)
    available_features = [col for col in df.columns if col != target] # The target variable chosen is removed from the features selectbox
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
        st.write('The Mean Squared Error tells you how far off the model predictions are from the actual values: larger errors are penalized more because the differences are squared. A lower value is generally better.')
        st.write(f'Root Mean Squared Error: {rmse_scaled:.2f}')
        st.write('The Root Mean Squared Error is similar to the MSE, but  it takes the square root, so it is in the same units as the target variable. A lower value is once again generally better.')
        st.write(f'R² Score: {r2_scaled:.2f}')
        st.write('The R² score tells you how much of the variance in the target variable the model is able to explain. This score is between 0 and 1, with higher scores generally being better.')

        st.header("\nModel Coefficients (Scaled):")
        st.write(pd.Series(lin_reg_scaled.coef_, index=features))
        st.header("\nModel Intercept (Scaled):")
        st.write(lin_reg_scaled.intercept_)

if model_option == 'Logistic Regression': # If the user chooses Logistic Regression

    st.sidebar.subheader('Select Features and Target for Logistic Regression')
    # Using a slider, the user has the option to set and adjust the test size for the train-test split
    test_size = st.sidebar.slider('Test Size for Train-Test Split', min_value=0.1, max_value=0.9, step=0.05, value=0.2)
    # Using a selectbox, the user has the option to choose the target and feature variables in the analysis
    target = st.sidebar.selectbox('Choose Target Variable', df.columns)
    available_features = [col for col in df.columns if col != target] # The target variable chosen is removed from the features selectbox
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

        # Generate and display the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.header("Confusion Matrix")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write('In logistic regression, the accuracy score tells you the percentage of correct predictions or classifications the model made, out of all predictions. A higher score is generally better.')
        st.pyplot(fig)
        st.write('In logistic regression, the confusion matrix provides a clear, detailed look at how well the model is classifying the data: not just whether the model is right or wrong, but how it is right or wrong. ' \
        ' This matrix is useful to identify which types of errors the model may be making by comparing predicted classifications to actual classifications.')

        # Display classification report
        report = classification_report(y_test, y_pred)
        st.header("Classification Report")
        st.markdown(f'```\n{report}\n```')

        st.write('The Classification Report provides the following assessments:')
        st.write('Precision: out of all predicted positives, how many were actually correct?')
        st.write('Recall: out of all actual positives, how many were actually correct?')
        st.write('F1-Score: a harmonic mean of precision and recall.')
        st.write('Support: the number of actual instances of each class in the test set.')

        # Check number of unique classes in the target variable
        unique_classes = y_test.nunique()

        if unique_classes == 2:
            # If there is binary classification in the target variable, generate and display the ROC curve
            try:
                y_probs = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)

                # Plot the ROC Curve
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax_roc.legend(loc="lower right")
                st.header("ROC Curve")
                st.pyplot(fig_roc)

                st.write('The ROC curve is a graph that shows how good the model is at distinguishing between two classes. Essentially, the ROC curve visualizes the tradeoff between recall and false positives at different thresholds.')

                # Display AUC Score
                st.subheader(f"ROC AUC Score: {roc_auc:.2f}")
                st.write('The AUC (Area Under the Curve) score provides a single scalar value to summarize the performance of the model. It ranges is from 0 to 1. A score of 1 signifies the model is a perfect classifier. A score of 0.5 means the model has no discriminative power (similar to guessing). A score less than 0.5 means the model may be confusing the two classes.')

            except Exception as e:
                st.error(f"Error computing ROC curve: {e}")
        else:
            # If the target variable is multiclass, the ROC curve is not generated nor displayed (avoids an error message as ROC curve is designed for binary classification)
            st.header("ROC Curve")
            st.write("ROC curve is only available for binary classification problems.")

if model_option == 'Decision Tree': # If the user chooses Decision Tree

    st.sidebar.subheader('Select Features and Target for Decision Tree')
    # Using a slider, the user has the option to set and adjust the test size for the train-test split
    test_size = st.sidebar.slider('Test Size for Train-Test Split', min_value=0.1, max_value=0.9, step=0.05, value=0.2)
    target = st.sidebar.selectbox('Choose Target Variable', df.columns)
    # Using a selectbox, the user has the option to choose the target and feature variables in the analysis
    available_features = [col for col in df.columns if col != target] # The target variable chosen is removed from the features selectbox
    features = st.sidebar.multiselect('Choose Features', available_features)

    # Create hyperparameter sliders in the sidebar to allow user to set and change hyperparameter values
    st.sidebar.subheader('Hyperparameter Tuning:')
    max_depth = st.sidebar.slider('Max Depth of Decision Tree', min_value=1, max_value=15, value=3)
    min_samples_split = st.sidebar.slider('Min Samples Split', min_value=2, max_value=20, value=2)
    min_samples_leaf = st.sidebar.slider('Min Samples Leaf', min_value=1, max_value=10, value=1)

    if target and features:
        # Prepare data for model
        X = df[features]
        y = df[target]

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        # Split the scaled data
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
 
        # Initialize decision tree model with custom hyperparameters from user-selected slider values and train model
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate and display the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        st.header("Confusion Matrix:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write('In a decision tree model, the accuracy score tells you the percentage of correct predictions or classifications the model made, out of all predictions. A higher score is generally better.')
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        st.write('In a decision tree model, the confusion matrix provides a clear, detailed look at how well the model is classifying the data: not just whether the model is right or wrong, but how it is right or wrong. ' \
        ' This matrix is useful to identify which types of errors the model may be making by comparing predicted classifications to actual classifications.')

        # Display classification report
        report = classification_report(y_test, y_pred)
        st.header("Classification Report")
        st.markdown(f'```\n{report}\n```')
        st.write('The Classification Report provides the following assessments:')
        st.write('Precision: out of all predicted positives, how many were actually correct?')
        st.write('Recall: out of all actual positives, how many were actually correct?')
        st.write('F1-Score: a harmonic mean of precision and recall.')
        st.write('Support: the number of actual instances of each class in the test set.')
        # Establish dynamic class names in the decision tree (makes the code more flexible across datasets)
        class_names = [str(cls) for cls in np.unique(y_train)]

        # Visualize the decision tree
        dot_data = tree.export_graphviz(
            model,
            feature_names=X_train_scaled.columns,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        st.header("Decision Tree")
        st.write('A decision tree visual illustrates how decisions are made based on the features of the dataset: specifically, it shows which features matter, and in what order they are used to split the data.' \
        ' Using yes-or-no questions at every node, the tree splits the data based on a feature that best separates the classes or reduces error.')
        st.graphviz_chart(dot_data)
        st.subheader("Hyperparameter Tuning")
        st.write('In the sidebar, there is a set of sliders that allow you for adjustments to the model hyperparameters. ' \
        ' These sliders provide an intuitive way to experiment with different configurations, like the maximum depth of the tree the minimum number of samples required to split an internal node, and the minimum number of samples that must be present in a leaf node.' \
        ' Adjusting these hyperparameters helps control the complexity of the model and discern a balance between underfitting and overfitting: the evaluation metrics dynamically change in response to hyperparameter tuning. ' \
        ' Check out the sidebar to try adjusting the hyperparameters to find the combination of hyperparameters that yields the best model performance!')
    
       # Check number of unique classes in the target variable
        unique_classes = y_test.nunique()

        if unique_classes == 2:
            # If there is binary classification in the target variable, generate and display the ROC curve
            try:
                y_probs = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)

                # Plot the ROC Curve
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax_roc.legend(loc="lower right")
                st.header("ROC Curve")
                st.pyplot(fig_roc)

                st.write('The ROC curve is a graph that shows how good the model is at distinguishing between two classes. Essentially, the ROC curve visualizes the tradeoff between recall and false positives at different thresholds.')

                # Display AUC Score
                st.subheader(f"ROC AUC Score: {roc_auc:.2f}")
                st.write('The AUC (Area Under the Curve) score provides a single scalar value to summarize the performance of the model. It ranges is from 0 to 1. A score of 1 signifies the model is a perfect classifier. A score of 0.5 means the model has no discriminative power (similar to guessing). A score less than 0.5 means the model may be confusing the two classes.')

            except Exception as e:
                st.error(f"Error computing ROC curve: {e}")
        else:
            # If the target variable is multiclass, the ROC curve is not generated nor displayed (avoids an error message as ROC curve is designed for binary classification)
            st.header("ROC Curve")
            st.write("ROC curve is only available for binary classification problems.")
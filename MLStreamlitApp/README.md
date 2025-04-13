# 📈 Machine Learning App Overview:
This interactive Streamlit app provides an intuitive platform for exploring datasets and applying supervised machine learning models. This app was designed with usability in mind and features dynamic menus and customizable widgets that enables users to tailor their analysis to specific needs or requirements.

With this app, users can:
- Upload their own datasets or choose from built-in sample datasets
- Select from a range of supervised learning models for both regression and classification
- Define and customize target and feature variables
- Visualize model performance with tools like confusion matrices, decision trees, and ROC curves

# 🚀 Instructions:

### Prerequisites
Ensure you have the following installed before running this app:
- Python (v. 3.12.7 recommended)
- streamlit (v. 1.44.1)
- pandas (v. 2.2.3)
- numpy (v. 2.2.3)
- scikit-learn (v. 1.6.1)
- matplotlib (v. 3.10.1)
- seaborn (v. 0.13.2)
- graphviz (v. 0.20.3)

### Running the Application

First, clone the DOLAN-Data-Science-Portfolio repository to create a local copy on your computer: clone the repository by downloading the repository as a ZIP file. Then, extract the downloaded ZIP file to reveal a folder containing all the repository project files. Next, navigate to the MLStreamlitApp folder within the extracted content, and upload this folder as your working directory. This folder should include the MLStreamlitApp.py file, as well as the README.md file.

To launch the application, use the following command in your terminal:

```bash
streamlit run MLStreamlitApp.py
```

The app should open automatically in your default web browser.

# ⚙️ App Features:

This machine learning app includes the following supervised learning machine models: Linear Regression, Logistic Regression, and Decision Trees. Below is a more detailed explanation of how each of these models are implemented within the app.

## Linear Regression

Within the Linear Regression model, users can customize key parameters, including:
- adjusting the test size for the train-test split via a slider
- selecting and modifying the target and feature variables for the regression from a drop-down menu

Once these options are configured, the app provides the scaled model evaluation metrics, scaled coefficients, and the scaled intercept.

## Logistic Regression

Within the Logistic Regression model, users can customize key parameters, including:
- adjusting the test size for the train-test split via a slider
- selecting and modifying the target and feature variables for the regression from a drop-down menu

Once these options are configured, the app provides the accuracy score, a classification report, and an AUC score. Furthermore, the app provides visualizations such as a confusion matrix and a ROC Curve.

## Decision Tree

Within the Decision Tree model, users can customize key parameters, including:
- adjusting the test size for the train-test split via a slider
- selecting and modifying the target and feature variables for the regression from a drop-down menu

Furthermore, within the Decision Tree model, users can set and customize key **hyperparameters** using sliders in the sidebar, including:
- the maximum depth of the tree
- the minimum number of samples required to split an internal node
- the minimum number of samples that must be present in a leaf node

Once these options are configured, the app provides the accuracy score, a classification report, and an AUC score. Furthermore, the app provides visualizations such as a confusion matrix, a decision tree, and a ROC Curve.

## Visualizations

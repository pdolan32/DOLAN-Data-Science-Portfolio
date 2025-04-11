# 📈 Machine Learning App
This machine learning app interactive Streamlit-based app for exploring datasets and applying supervised machine learning models. This project aims to offer users an intuitive interface that makes it easy to apply supervised machine learning models to datasets efficiently: interactive menus and customizable widgets allow users to tailor their experience to fit their specific analytical needs.

# 🚀 Project Overview:
This project allows users to:
- Upload their own dataset or choose from built-in sample datasets.
- Select from a variety of supervised machine learning models (both regression and classification)
- Choose the target and feature variables.
- Visualize model performance with 

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



# App Features:

This machine learning app includes Linear Regression, Logistic Regression, and Decision Tree models.

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

## Visualizations

Once these options are configured, the app provides the accuracy score, a classification report, and an AUC score. Furthermore, the app provides visualizations such as a confusion matrix, a decision tree, and a ROC Curve.

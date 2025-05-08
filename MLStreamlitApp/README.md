# Supervised Machine Learning App Overview:
This interactive Streamlit app offers an intuitive platform for exploring datasets and applying supervised machine learning models. The app features dynamic menus and customizable widgets that allow users to tailor their analysis to specific needs and preferences.

The goal of this project is to build an accessible, user-friendly environment that empowers individuals, regardless of technical expertise, to engage with supervised machine learning for both classification and regression tasks: essentially, the app aims to simplify machine learning workflows, support data-driven decision-making, and foster experimentation and learning.

With this app, users can:
- Upload their own datasets or choose from built-in sample datasets
- Select from a range of supervised learning models for both regression and classification
- Specify and customize target and feature variables, along with model hyperparameters, to fine-tune model analysis and performance
- Visualize model performance with tools like confusion matrices, decision trees, and ROC curves

Click [HERE](https://dolan-data-science-portfolio-supervisedml-app.streamlit.app/) to access the app online.

# Instructions:

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

# App Features:

This machine learning app includes the following supervised learning machine models: Linear Regression, Logistic Regression, and Decision Trees. Below is a more detailed explanation of how each of these models are implemented within the app.

## Linear Regression

**Linear Regression** predicts a continuous numeric value by modeling the relationship between input features (X) and a continuous output (y) by fitting a straight line.

Within the Linear Regression model, users can customize key parameters, including:
- adjusting the test size for the train-test split via a slider
- selecting and modifying the target and feature variables for the regression from a drop-down menu

Once these options are configured, the app provides the scaled model evaluation metrics, scaled coefficients, and the scaled intercept.

## Logistic Regression

**Logistic Regression** predicts a categorical outcome using a logistic function to model the probability that a given input belongs to a particular class.

Within the Logistic Regression model, users can customize key parameters, including:
- adjusting the test size for the train-test split via a slider
- selecting and modifying the target and feature variables for the regression from a drop-down menu

Once these options are configured, the app provides the accuracy score, a classification report, and an AUC score. Furthermore, the app provides visualizations such as a confusion matrix and a ROC Curve.

## Decision Tree

A **decision tree** model splits the data into subsets based on feature values, using conditions to create a tree structure of decisions and ultimately classify data.

Within the Decision Tree model, users can customize key parameters, including:
- adjusting the test size for the train-test split via a slider
- selecting and modifying the target and feature variables for the regression from a drop-down menu

Furthermore, within the Decision Tree model, users can set and customize key **hyperparameters** using sliders in the sidebar, including:
- the maximum depth of the tree
- the minimum number of samples required to split an internal node
- the minimum number of samples that must be present in a leaf node

Once these options are configured, the app provides the accuracy score, a classification report, and an AUC score. Furthermore, the app provides visualizations such as a confusion matrix, a decision tree, and a ROC Curve.

## References

Grokking Machine Learning Chapter 7: Classification Models. Click [HERE](https://github.com/pdolan32/DOLAN-Data-Science-Portfolio/blob/main/MLStreamlitApp/GrokkingML_Measuring%20Classification%20Models-1.pdf) to view this reference.

Grokking Machine Learning Chapter 9: Decision Trees. Click [HERE](https://github.com/pdolan32/DOLAN-Data-Science-Portfolio/blob/main/MLStreamlitApp/GrokkingML_Decision%20Trees.pdf) to view this reference.

Streamlit Website: Input Widgets. Click [HERE](https://docs.streamlit.io/develop/api-reference/widgets) to access this reference (external website).

Scikit-Learn: Toy Datasets. Click [HERE](https://scikit-learn.org/stable/datasets/toy_dataset.html) to access this reference (external website).

## Visualizations

#### Here are some examples of the visualizations and reports produced by the app when the user chooses to analyze a dataset using the 'Decision Tree' model.

<img width="600" alt="MLApp_CM" src="https://github.com/user-attachments/assets/0baebbdb-1053-4805-b3b8-bb6ac091bad6" />

<img width="600" alt="MLApp_CR" src="https://github.com/user-attachments/assets/79ea0f77-d1c6-4a3b-9518-884162b2ca83" />

<img width="600" alt="MLApp_Tree" src="https://github.com/user-attachments/assets/dd84593f-cbbe-42f0-ab3b-303fe33b538e" />

<img width="600" alt="MLApp_ROC" src="https://github.com/user-attachments/assets/1c22b1d4-1aab-4938-85ea-57aac67a7db8" />







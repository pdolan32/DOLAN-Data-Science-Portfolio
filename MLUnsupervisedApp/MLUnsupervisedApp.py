# Import the required libraries
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes, load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Write in the title of the app
st.title('Unsupervised Machine Learning Dataset Analysis')

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
    dataset_option = st.sidebar.selectbox('Choose Sample Dataset', ('Breast Cancer'))

    if dataset_option == 'Breast Cancer': # If the user chooses the Breast Cancer Dataset
        # Description of the dataset
        st.write('This is the Breast Cancer dataset.' \
        ' This dataset is a well-known dataset used for binary classification, often to test models that distinguish between benign and malignant tumors: The target value is the diagnosis, with 0 = malignant and 1 = benign.')
        cancer = load_breast_cancer() # load in the dataset from sklearn.datasets
        X = cancer.data  # Feature matrix
        y = cancer.target  # Target variable (diagnosis)
        feature_names = cancer.feature_names
        target_names = cancer.target_names
        df = pd.DataFrame(X, columns = [feature_names])
        st.header("Sample Dataset: Breast Cancer")
        st.write(df.head()) # The first five rows of the dataset are displayed
        st.subheader("Summary Statistics:")
        st.write(df.describe()) # Summary statistics and information are displayed
        # Further instructions are suggested
        st.write('To analyze this dataset, please choose a supervised machine learning model from the sidebar.' \
        ' This dataset works best with classification models (like Logistic Regression and Decision Trees).')

# In the sidebar, the user is also presented with a selectbox to choose the supervised learning model they would like to apply to the dataset
model_option = st.sidebar.selectbox('Choose a Unsupervised Machine Learning Model', ('None', 'K-Means Clustering', 'Hierarchical Clustering', 'Principal Component Analysis'))

if model_option == 'Principal Component Analysis': 

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    # Display the Explained Variance Ratio (i.e., the proportion of variance explained by each component)
    explained_variance = pca.explained_variance_ratio_
    print("Explained Variance Ratio:", explained_variance)
    print("Cumulative Explained Variance:", np.cumsum(explained_variance))

    # Scatter Plot of PCA Scores
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    colors = ['navy', 'darkorange']
    for color, i, target_name in zip(colors, [0, 1], target_names):
        ax1.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.7,
                    label=target_name, edgecolor='k', s=60)
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_title('PCA: 2D Projection of Breast Cancer Data')
    ax1.legend(loc='best')
    ax1.grid(True)
    st.pyplot(fig1)

    # Biplot: Overlaying Feature Loadings on PCA Scatter Plot
    # Compute the loadings (each column of pca.components_ represents a principal component)
    loadings = pca.components_.T
    scaling_factor = 50.0

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for color, i, target_name in zip(colors, [0, 1], target_names):
        ax2.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.7,
                    label=target_name, edgecolor='k', s=60)
    for i, feature in enumerate(feature_names):
        ax2.arrow(0, 0, scaling_factor * loadings[i, 0], scaling_factor * loadings[i, 1],
                  color='r', width=0.02, head_width=0.1)
        ax2.text(scaling_factor * loadings[i, 0] * 1.1, scaling_factor * loadings[i, 1] * 1.1,
                 feature, color='r', ha='center', va='center')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.set_title('Biplot: PCA Scores and Loadings')
    ax2.legend(loc='best')
    ax2.grid(True)
    st.pyplot(fig2)

    # Scree Plot: Cumulative Explained Variance
    # This plot helps in determining how many components to retain (looking for the "elbow")
    pca_full = PCA(n_components=min(X_std.shape[1], 15)).fit(X_std)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
    ax3.set_xlabel('Number of Components')
    ax3.set_ylabel('Cumulative Explained Variance')
    ax3.set_title('PCA Variance Explained')
    ax3.set_xticks(range(1, len(cumulative_variance)+1))
    ax3.grid(True)
    st.pyplot(fig3)

    # Bar Plot: Variance Explained by Each Component

    fig, ax = plt.subplots(figsize=(8, 6))
    components = range(1, len(pca_full.explained_variance_ratio_) + 1)
    ax.bar(components, pca_full.explained_variance_ratio_, alpha=0.7, color='teal')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('Variance Explained by Each Principal Component')
    ax.set_xticks(components)
    ax.grid(True, axis='y')

    st.pyplot(fig)

    explained = pca_full.explained_variance_ratio_ * 100
    components = np.arange(1, len(explained) + 1)
    cumulative = np.cumsum(explained)

    # Combined Plot
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    bar_color = 'steelblue'
    ax4.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Individual Variance Explained (%)', color=bar_color)
    ax4.tick_params(axis='y', labelcolor=bar_color)
    ax4.set_xticks(components)
    ax4.set_xticklabels([f"PC{i}" for i in components])
    for i, v in enumerate(explained):
        ax4.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

    ax5 = ax4.twinx()
    line_color = 'crimson'
    ax5.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance')
    ax5.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
    ax5.tick_params(axis='y', labelcolor=line_color)
    ax5.set_ylim(0, 100)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax5.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

    plt.title('PCA: Variance Explained', pad=20)
    st.pyplot(fig4)

    # Split the standardized (original) data into training and test sets
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

    # Split the PCA-reduced data into training and test sets (using the same random state)
    X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # 4a. Logistic Regression on Original Data
    clf_orig = LogisticRegression()
    clf_orig.fit(X_train_orig, y_train)
    y_pred_orig = clf_orig.predict(X_test_orig)
    acc_orig = accuracy_score(y_test, y_pred_orig)
    st.write("Logistic Regression Accuracy on Original Data: {:.2f}%".format(acc_orig * 100))

    # 4a. Logistic Regression on PCA Data
    clf_pca = LogisticRegression()
    clf_pca.fit(X_train_pca, y_train)
    y_pred_pca = clf_pca.predict(X_test_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    st.write("Logistic Regression Accuracy on PCA: {:.2f}%".format(acc_pca * 100))

if model_option == 'K-Means Clustering': 

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    k = 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_std)

   # Reduce the data to 2 dimensions for visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    # Create a scatter plot of the PCA-transformed data, colored by KMeans cluster labels
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(X_pca[clusters == 0, 0], X_pca[clusters == 0, 1],
                c='navy', alpha=0.7, edgecolor='k', s=60, label='Cluster 0')
    ax1.scatter(X_pca[clusters == 1, 0], X_pca[clusters == 1, 1],
                c='darkorange', alpha=0.7, edgecolor='k', s=60, label='Cluster 1')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_title('KMeans Clustering: 2D PCA Projection')
    ax1.legend(loc='best')
    ax1.grid(True)
    st.pyplot(fig1)

    # For comparison, visualize true labels using PCA (same 2D projection)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    colors = ['navy', 'darkorange']
    for i, target_name in enumerate(target_names):
        ax2.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                    color=colors[i], alpha=0.7, edgecolor='k', s=60, label=target_name)
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.set_title('True Labels: 2D PCA Projection')
    ax2.legend(loc='best')
    ax2.grid(True)
    st.pyplot(fig2)

    from sklearn.metrics import accuracy_score

    # Since KMeans labels are arbitrary (e.g., 0 and 1) and may not match the true labels directly,
    # we compute accuracy for both the original labels and their complement, and choose the higher value.

    kmeans_accuracy = accuracy_score(y, clusters)

    print("Accuracy Score: {:.2f}%".format(kmeans_accuracy * 100))

    # Define the range of k values to try
    ks = range(2, 11)
    wcss = []
    silhouette_scores = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_std)
        wcss.append(km.inertia_)
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X_std, labels))

    st.write("Within-Cluster Sum of Squares (WCSS):", wcss)
    st.write("Silhouette Scores:", silhouette_scores)

    # Plot the Elbow Method result
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    ax3a.plot(ks, wcss, marker='o')
    ax3a.set_xlabel('Number of Clusters (k)')
    ax3a.set_ylabel('WCSS')
    ax3a.set_title('Elbow Method for Optimal k')
    ax3a.grid(True)

    ax3b.plot(ks, silhouette_scores, marker='o', color='green')
    ax3b.set_xlabel('Number of Clusters (k)')
    ax3b.set_ylabel('Silhouette Score')
    ax3b.set_title('Silhouette Score for Optimal k')
    ax3b.grid(True)

    plt.tight_layout()
    st.pyplot(fig3)
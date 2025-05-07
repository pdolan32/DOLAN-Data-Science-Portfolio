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
from scipy.cluster.hierarchy import linkage, dendrogram
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

target_column = None
df = None

if option == 'Upload Your Own': # If the user chooses to upload their own dataset
    
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.write(df.head())
        st.subheader("Summary Statistics:")
        st.write(df.describe())

        # Optionally allow user to select a target column
        if st.sidebar.checkbox("Does your dataset include a target column?"):
            target_column = st.sidebar.selectbox("Select target column:", df.columns)

else:
    dataset_option = st.sidebar.selectbox('Choose Sample Dataset', ('Breast Cancer', 'Iris'))

    if dataset_option == 'Breast Cancer':
        data = load_breast_cancer()
    elif dataset_option == 'Iris':
        data = load_iris()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    if hasattr(data, 'target'):
        df['target'] = data.target
        target_column = 'target'
        target_names = data.target_names if hasattr(data, 'target_names') else np.unique(data.target)
    else:
        target_names = None

    st.write(f"Sample Dataset: {dataset_option}")
    st.write(df.head())
    st.subheader("Summary Statistics:")
    st.write(df.describe())

if df is not None:
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column].values
    else:
        X = df.copy()
        y = None
        target_names = None

    # Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    model_option = st.sidebar.selectbox('Choose a Model', ('None', 'Principal Component Analysis', 'K-Means Clustering'))

    if model_option == 'Principal Component Analysis':
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)

        explained_variance = pca.explained_variance_ratio_
        st.write("Explained Variance Ratio:", explained_variance)
        st.write("Cumulative Explained Variance:", np.cumsum(explained_variance))

        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"Feature {i}" for i in range(X.shape[1])]
        
        unique_labels = np.unique(y) if y is not None else []
        n_classes = len(unique_labels)
        color_map = plt.cm.get_cmap('tab10', n_classes)

        # --- 1. PCA 2D Scatter Plot ---
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        if y is not None:
            for i, label in enumerate(unique_labels):
                label_name = str(target_names[label]) if target_names is not None else str(label)
                ax1.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                            color=color_map(i), alpha=0.7, edgecolor='k', s=60, label=label_name)
        else:
            ax1.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.7, edgecolor='k', s=60)
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('PCA: 2D Projection')
        ax1.legend(loc='best')
        ax1.grid(True)
        st.pyplot(fig1)

        # --- 2. PCA Biplot with Loadings ---
        loadings = pca.components_.T
        scaling_factor = 50.0
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        if y is not None:
            for i, label in enumerate(unique_labels):
                label_name = str(target_names[label]) if target_names is not None else str(label)
                ax2.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                            color=color_map(i), alpha=0.7, edgecolor='k', s=60, label=label_name)
        else:
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.7, edgecolor='k', s=60)
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

        # --- 3. Scree Plot of Cumulative Explained Variance ---
        pca_full = PCA(n_components=min(X_std.shape[1], 15)).fit(X_std)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Cumulative Explained Variance')
        ax3.set_title('PCA Variance Explained')
        ax3.set_xticks(range(1, len(cumulative_variance) + 1))
        ax3.grid(True)
        st.pyplot(fig3)

        # --- 4. Bar Plot of Individual Variance ---
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        components = range(1, len(pca_full.explained_variance_ratio_) + 1)
        ax4.bar(components, pca_full.explained_variance_ratio_, alpha=0.7, color='teal')
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Variance Explained')
        ax4.set_title('Variance Explained by Each Principal Component')
        ax4.set_xticks(components)
        ax4.grid(True, axis='y')
        st.pyplot(fig4)

        # --- 5. Combined Bar + Cumulative Line Plot ---
        explained = pca_full.explained_variance_ratio_ * 100
        components = np.arange(1, len(explained) + 1)
        cumulative = np.cumsum(explained)

        fig5, ax5 = plt.subplots(figsize=(8, 6))
        bar_color = 'steelblue'
        ax5.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')
        ax5.set_xlabel('Principal Component')
        ax5.set_ylabel('Individual Variance Explained (%)', color=bar_color)
        ax5.tick_params(axis='y', labelcolor=bar_color)
        ax5.set_xticks(components)
        ax5.set_xticklabels([f"PC{i}" for i in components])
        for i, v in enumerate(explained):
            ax5.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

        ax6 = ax5.twinx()
        line_color = 'crimson'
        ax6.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance')
        ax6.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
        ax6.tick_params(axis='y', labelcolor=line_color)
        ax6.set_ylim(0, 100)

        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax6.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

        plt.title('PCA: Variance Explained', pad=20)
        st.pyplot(fig5)

        # --- 6. Optional Logistic Regression Evaluation (only if labels exist) ---
        if y is not None:
            X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
            X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)

            clf_orig = LogisticRegression()
            clf_orig.fit(X_train_orig, y_train)
            y_pred_orig = clf_orig.predict(X_test_orig)
            acc_orig = accuracy_score(y_test, y_pred_orig)
            st.write("Logistic Regression Accuracy on Original Data: {:.2f}%".format(acc_orig * 100))

            clf_pca = LogisticRegression()
            clf_pca.fit(X_train_pca, y_train)
            y_pred_pca = clf_pca.predict(X_test_pca)
            acc_pca = accuracy_score(y_test, y_pred_pca)
            st.write("Logistic Regression Accuracy on PCA Data: {:.2f}%".format(acc_pca * 100))

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
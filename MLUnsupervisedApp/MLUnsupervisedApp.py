# Import the required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
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
    dataset_option = st.sidebar.selectbox('Choose Sample Dataset', ('Breast Cancer', 'Iris', 'Wine'))

    if dataset_option == 'Breast Cancer':
        data = load_breast_cancer()
    elif dataset_option == 'Iris':
        data = load_iris()
    elif dataset_option == 'Wine':
        data= load_wine()

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

    model_option = st.sidebar.selectbox('Choose a Model', ('None', 'Principal Component Analysis', 'K-Means Clustering', 'Hierarchical Clustering'))

    if model_option == 'Principal Component Analysis':
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        max_components = min(X.shape[1], 15)  # Limit to 15 or total number of features
        n_components = st.sidebar.slider('Number of Principal Components', 2, max_components, 2)

        # The data is reduced to 2 components for visualization purposes and further analysis.
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_std)

        # Display the Explained Variance Ratio (i.e., the proportion of variance explained by each component)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # Format as percentages with 2 decimal places
        explained_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
            'Explained Variance (%)': (explained_variance * 100).round(2),
            'Cumulative Variance (%)': (cumulative_variance * 100).round(2)
        }, index=np.arange(1, len(explained_variance) + 1))

        # Display the table in Streamlit
        st.subheader("PCA Explained Variance")
        st.dataframe(explained_df)

        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"Feature {i}" for i in range(X.shape[1])]
        
        unique_labels = np.unique(y) if y is not None else []
        n_classes = len(unique_labels)
        color_map = plt.cm.get_cmap('tab10', n_classes)

        # --- 1. PCA 2D Scatter Plot ---
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        color_map = plt.get_cmap('tab10')
        if y is not None: # Checks if a target variable y (e.g., class labels) is available. If so, plot points with different colors based on their class.
            for i, label in enumerate(unique_labels): # Loops through each unique class label.
                label_name = str(target_names[label]) if target_names is not None else str(label)
                ax1.scatter(X_pca[y == label, 0], X_pca[y == label, 1], # Plots the PCA points that belong to a specific class
                            color=color_map(i), alpha=0.7, edgecolor='k', s=60, label=label_name)
        else: # If no y (target labels), it plots all points in gray without label-based separation.
            ax1.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.7, edgecolor='k', s=60)
        # Below code adds axis labels and a title to the plot.
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('PCA: 2D Projection')
        ax1.legend(loc='best') # Adds a legend with class names (only if y is provided).
        ax1.grid(True) # Adds a grid to make the plot easier to read.
        st.pyplot(fig1) # Renders the figure in the Streamlit app.

        # --- 2. PCA Biplot with Feature Loadings ---
        loadings = pca.components_.T # Retrieves the loadings (a.k.a. component weights), which describe the contribution of each original feature to each principal component.
        scaling_factor = 50.0
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        color_map = plt.get_cmap('tab10')
        if y is not None: # If class labels (y) are available: plots the PCA-transformed data, coloring each class differently.
            for i, label in enumerate(unique_labels):
                label_name = str(target_names[label]) if target_names is not None else str(label)
                ax2.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                            color=color_map(i), alpha=0.7, edgecolor='k', s=60, label=label_name)
        else: # If no labels are provided, it plots all points in gray.
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.7, edgecolor='k', s=60)
        for i, feature in enumerate(feature_names): # Loops through each original feature to plot its vector (arrow) on the PCA plot.
            ax2.arrow(0, 0, scaling_factor * loadings[i, 0], scaling_factor * loadings[i, 1], # Draws an arrow from the origin (0, 0) to the scaled coordinates in PCA space.
                    color='r', width=0.02, head_width=0.1)
            ax2.text(scaling_factor * loadings[i, 0] * 1.1, scaling_factor * loadings[i, 1] * 1.1, # Adds a text label for each arrow at the tip.
                    feature, color='r', ha='center', va='center')
        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.set_title('Biplot: PCA Scores and Feature Loadings')
        ax2.legend(loc='best') # Adds a legend if y is present (class labels).
        ax2.grid(True)
        st.pyplot(fig2) # Renders the figure in the Streamlit app.

        # --- 3. Scree Plot of Cumulative Explained Variance ---
        pca_full = PCA(n_components=n_components).fit(X_std) # Creates and fits a PCA model using up to the specified amount of components)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_) # Calculates the cumulative sum of the explained variance ratio
        fig3, ax3 = plt.subplots(figsize=(8, 6)) # Creates a new figure and axis for the line chart.
        ax3.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o') # Plots the cumulative variance explained versus the number of components.
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Cumulative Explained Variance')
        ax3.set_title('PCA Variance Explained')
        ax3.set_xticks(range(1, len(cumulative_variance) + 1)) # Ensures the x-axis only shows integer component counts (e.g., 1 to 15).
        ax3.grid(True)
        st.pyplot(fig3) # Renders the plot within the Streamlit app

        # --- 4. Bar Plot of Individual Variance ---
        fig4, ax4 = plt.subplots(figsize=(8, 6)) # Creates a new figure and axis for the bar plot.
        components = range(1, len(pca_full.explained_variance_ratio_) + 1) # Creates a range of component numbers (1-based) to use as x-axis labels.
        ax4.bar(components, pca_full.explained_variance_ratio_, alpha=0.7, color='teal') # Draws a bar for each principal component, where the height represents how much variance that component explains.
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Variance Explained')
        ax4.set_title('Variance Explained by Each Principal Component')
        ax4.set_xticks(components) # Ensures x-axis ticks match the component numbers exactly.
        ax4.grid(True, axis='y')
        st.pyplot(fig4) # Renders the chart within the Streamlit app

        # --- 5. Combined Bar + Cumulative Line Plot ---
        explained = pca_full.explained_variance_ratio_ * 100 # Converts the explained variance from a ratio to a percentage.
        components = np.arange(1, len(explained) + 1) # Generates component numbers starting at 1 (for display and labeling).
        cumulative = np.cumsum(explained) # Calculates the cumulative percentage of variance explained.

        fig5, ax5 = plt.subplots(figsize=(8, 6)) # Creates the primary axis and draws a bar for each component showing the % of variance it individually explains.
        bar_color = 'steelblue'
        # Labels the x-axis and left y-axis, matching their colors for clarity.
        ax5.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')
        ax5.set_xlabel('Principal Component')
        ax5.set_ylabel('Individual Variance Explained (%)', color=bar_color)
        ax5.tick_params(axis='y', labelcolor=bar_color)
        ax5.set_xticks(components)
        ax5.set_xticklabels([f"PC{i}" for i in components])
        for i, v in enumerate(explained): # Adds text labels above each bar showing the exact variance %.
            ax5.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

        ax6 = ax5.twinx() # Adds a second y-axis that shares the same x-axis, and plots the cumulative variance as a line.
        line_color = 'crimson'
        # Sets and styles the right y-axis, ensuring it scales from 0% to 100%.
        ax6.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance')
        ax6.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
        ax6.tick_params(axis='y', labelcolor=line_color)
        ax6.set_ylim(0, 100)

        # Combines both legends (from bar and line plots) into one unified legend placed neatly on the side.
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax6.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

        # Adds a descriptive title with spacing above and renders the final chart in Streamlit.
        plt.title('PCA: Variance Explained', pad=20)
        st.pyplot(fig5)

    if model_option == 'K-Means Clustering': 

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Set number of clusters (you can later make this a sidebar input)
        k = st.sidebar.slider('Select number of clusters (k)', min_value=2, max_value=10, value=2, step=1)
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_std)

        # Reduce to 2D with PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)

        # --- Plot 1: KMeans Cluster Labels in PCA Space ---
        fig1, ax1 = plt.subplots(figsize=(8, 6))

        cluster_labels = np.unique(clusters)
        n_clusters = len(cluster_labels)
        # Use tab10 for up to 10 clusters, tab20 otherwise
        color_map = plt.cm.get_cmap('tab10')

        for i in cluster_labels:
            ax1.scatter(
                X_pca[clusters == i, 0], X_pca[clusters == i, 1],
                color=color_map(i), alpha=0.7, edgecolor='k', s=60,
                label=f'Cluster {i}'
            )

        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('KMeans Clustering: 2D PCA Projection')
        ax1.legend(loc='best')
        ax1.grid(True)
        st.pyplot(fig1)

        # --- Plot 2: True Labels (if provided) in PCA Space ---
        fig2, ax2 = plt.subplots(figsize=(8, 6))

        if y is not None:
            true_labels = np.unique(y)
            n_classes = len(true_labels)
            color_map = plt.cm.get_cmap('tab10')

            for i in true_labels:
                label_name = str(target_names[i]) if target_names is not None else str(i)
                ax2.scatter(
                    X_pca[y == i, 0], X_pca[y == i, 1],
                    color=color_map(i), alpha=0.7, edgecolor='k', s=60,
                    label=label_name
                )
            ax2.set_title('True Labels: 2D PCA Projection')
        else:
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.7, edgecolor='k', s=60)
            ax2.set_title('True Labels (Unavailable)')

        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.legend(loc='best')
        ax2.grid(True)
        st.pyplot(fig2)

        # --- Accuracy Score ---
        kmeans_accuracy = accuracy_score(y, clusters)

        st.write("Accuracy Score: {:.2f}%".format(kmeans_accuracy * 100))

        # --- Elbow Method + Silhouette Scores ---
        ks = range(2, 11)
        wcss = []
        silhouette_scores = []

        for k in ks:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X_std)
            wcss.append(km.inertia_)
            labels = km.labels_
            silhouette_scores.append(silhouette_score(X_std, labels))

        # Display metrics in a DataFrame
        metrics_df = pd.DataFrame({
            'WCSS': wcss,
            'Silhouette Score': silhouette_scores
        }, index=ks)
        metrics_df.index.name = "k"  # Label the index for clarity

        st.dataframe(metrics_df.style.format({"WCSS": "{:.2f}", "Silhouette Score": "{:.3f}"}))

        # --- Plot 3: Elbow Method and Silhouette Scores ---
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

    if model_option == 'Hierarchical Clustering': 
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- Sidebar Controls ---
        linkage_option = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
        k = st.sidebar.slider('Select number of clusters (k)', min_value=2, max_value=10, value=2, step=1)

        # --- Dendrogram ---
        Z = linkage(X_scaled, method=linkage_option)

        if y is not None:
            labels = y.astype(str).tolist()
        else:
            labels = df.index.astype(str).tolist()

        st.write("### Dendrogram")
        fig1, ax1 = plt.subplots(figsize=(20, 7))
        dendrogram(Z, labels=labels, ax=ax1)
        ax1.set_title("Hierarchical Clustering Dendrogram")
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("Distance")
        st.pyplot(fig1)

        # --- Silhouette Score Curve ---
        st.write("### Silhouette Score Analysis")

        k_range = range(2, 11)
        sil_scores = []

        for k_test in k_range:
            temp_labels = AgglomerativeClustering(n_clusters=k_test, linkage=linkage_option).fit_predict(X_scaled)
            score = silhouette_score(X_scaled, temp_labels)
            sil_scores.append(score)

        best_k = k_range[np.argmax(sil_scores)]

        fig_sil, ax_sil = plt.subplots(figsize=(7, 4))
        ax_sil.plot(list(k_range), sil_scores, marker="o")
        ax_sil.set_xticks(list(k_range))
        ax_sil.set_xlabel("Number of Clusters (k)")
        ax_sil.set_ylabel("Average Silhouette Score")
        ax_sil.set_title("Silhouette Analysis for Agglomerative Clustering")
        ax_sil.grid(True, alpha=0.3)
        st.pyplot(fig_sil)

        st.write(f"**Best k by silhouette score: {best_k}**  _(score = {max(sil_scores):.3f})_")

        # --- Final Clustering with User-Selected k ---
        agg = AgglomerativeClustering(n_clusters=k, linkage=linkage_option)
        cluster_labels = agg.fit_predict(X_scaled)

        clustered_df = df.copy()
        clustered_df["Cluster"] = cluster_labels

        # --- Clustered Table Preview ---
        st.write("### Cluster Assignments (First 10 Rows)")
        st.dataframe(clustered_df.head(10))

        st.write("### Cluster Sizes")
        st.dataframe(clustered_df["Cluster"].value_counts().reset_index().rename(columns={"index": "Cluster", "Cluster": "Count"}))

        # --- PCA Visualization ---
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.write("### Cluster Visualization (PCA Reduced)")
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=60, edgecolor='k', alpha=0.7)
        ax2.set_xlabel("Principal Component 1")
        ax2.set_ylabel("Principal Component 2")
        ax2.set_title(f"Agglomerative Clustering (PCA View, k={k}, linkage='{linkage_option}')")
        ax2.legend(*scatter.legend_elements(), title="Clusters")
        ax2.grid(True)
        st.pyplot(fig2)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="AI E-commerce Customer Behavior Analyzer",
    layout="wide"
)

st.title("AI E-commerce Customer Behavior Analyzer")

uploaded_file = st.file_uploader(
    "Upload Pakistan E-commerce Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Data Cleaning")

    df = df.drop_duplicates()

    df = df.dropna(subset=['Customer ID'])

    df = df.fillna("Unknown")

    df['created_at'] = pd.to_datetime(df['created_at'])

    df['grand_total'] = pd.to_numeric(
        df['grand_total'],
        errors='coerce'
    )

    st.success("Data Cleaning Completed")

    st.subheader("Order Status Analysis")

    fig1, ax1 = plt.subplots(figsize=(10,5))

    sns.countplot(
        x=df['status'],
        ax=ax1
    )

    plt.xticks(rotation=45)

    st.pyplot(fig1)

    st.subheader("Top Product Categories")

    top_categories = df['category_name_1'].value_counts().head(10)

    fig2, ax2 = plt.subplots(figsize=(12,5))

    sns.barplot(
        x=top_categories.index,
        y=top_categories.values,
        ax=ax2
    )

    plt.xticks(rotation=45)

    st.pyplot(fig2)

    st.subheader("Payment Methods")

    fig3, ax3 = plt.subplots(figsize=(10,5))

    sns.countplot(
        x=df['payment_method'],
        ax=ax3
    )

    plt.xticks(rotation=45)

    st.pyplot(fig3)

    st.subheader("RFM Analysis")

    latest_date = df['created_at'].max()

    rfm = df.groupby('Customer ID').agg({
        'created_at': lambda x: (latest_date - x.max()).days,
        'increment_id': 'count',
        'grand_total': 'sum'
    })

    rfm.columns = [
        'Recency',
        'Frequency',
        'Monetary'
    ]

    st.dataframe(rfm.head())

    st.subheader("Data Scaling")

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(rfm)

    st.success("Scaling Completed")

    st.subheader("Elbow Method")

    wcss = []

    for i in range(1,11):

        kmeans = KMeans(
            n_clusters=i,
            random_state=42
        )

        kmeans.fit(rfm_scaled)

        wcss.append(kmeans.inertia_)

    fig4, ax4 = plt.subplots(figsize=(8,5))

    ax4.plot(range(1,11), wcss, marker='o')

    ax4.set_xlabel("Clusters")
    ax4.set_ylabel("WCSS")
    ax4.set_title("Elbow Method")

    st.pyplot(fig4)

    st.subheader("K-Means Clustering")

    cluster_number = st.slider(
        "Select Number of Clusters",
        2,
        10,
        4
    )

    kmeans = KMeans(
        n_clusters=cluster_number,
        random_state=42
    )

    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    st.dataframe(rfm.head())

    st.subheader("Cluster Distribution")

    fig5, ax5 = plt.subplots(figsize=(8,5))

    sns.countplot(
        x=rfm['Cluster'],
        ax=ax5
    )

    st.pyplot(fig5)

    st.subheader("PCA Visualization")

    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(rfm_scaled)

    pca_df = pd.DataFrame(
        pca_data,
        columns=['PC1', 'PC2']
    )

    pca_df['Cluster'] = rfm['Cluster'].values

    fig6, ax6 = plt.subplots(figsize=(10,6))

    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue='Cluster',
        data=pca_df,
        palette='Set1',
        ax=ax6
    )

    st.pyplot(fig6)

    st.subheader("Cluster Analysis")

    cluster_analysis = rfm.groupby('Cluster').mean()

    st.dataframe(cluster_analysis)

    st.subheader("Heatmap")

    fig7, ax7 = plt.subplots(figsize=(10,5))

    sns.heatmap(
        cluster_analysis,
        annot=True,
        cmap='coolwarm',
        ax=ax7
    )

    st.pyplot(fig7)

    st.subheader("Business Insights")

    st.markdown("""
    ### High Value Customers
    - High spending customers
    - Frequent buyers

    ### Low Value Customers
    - Low spending customers
    - Need promotions

    ### At Risk Customers
    - High recency
    - Low frequency
    """)

    st.subheader("Revenue by Cluster")

    cluster_revenue = rfm.groupby(
        'Cluster'
    )['Monetary'].sum()

    fig8, ax8 = plt.subplots(figsize=(8,5))

    sns.barplot(
        x=cluster_revenue.index,
        y=cluster_revenue.values,
        ax=ax8
    )

    st.pyplot(fig8)

    st.success("Project Completed Successfully")
    



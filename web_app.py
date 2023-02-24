#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt

# to perform k-means clustering and compute silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA
# to visualize the elbow curve and silhouette scores
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# to suppress warnings
import warnings
warnings.filterwarnings("ignore")


from PIL import Image

import csv

from sklearn.model_selection import cross_val_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:





# In[ ]:





# In[23]:


# Load the CSV file
with open('./product_images.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the first row
    pixel_data = list(reader)

data = pd.read_csv("./product_images.csv")


# In[3]:



# Convert the pixel data to a numpy array
pixel_array = np.array(pixel_data, dtype=np.uint8)

# Reshape the pixel data to the dimensions of the original images
pixel_array = pixel_array.reshape((len(pixel_array), 28, 28))



# In[4]:


# Reshape the pixel array to have one row for each image
X = pixel_array.reshape(len(pixel_array), -1)

# Determine the optimal number of components using the cumulative explained variance plot
pca = PCA()
pca.fit(X)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)


# Choose the number of components that capture at least 20% of the variance
n_components = np.argmax(cumulative_variance >= 0.3) + 1
print(f"Number of components chosen: {n_components}")


# In[5]:




# Perform PCA to reduce the dimensionality of the data
pca = PCA(n_components)
X_pca = pca.fit_transform(pixel_data)

# Initialize empty lists to store inertia and silhouette scores for different values of K
inertias = []
silhouette_scores = []

# Define range of K values to try
K_values = range(2, 30)

# Loop over each K value
for K in K_values:
    # Fit K-means model to the PCA-transformed data
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X_pca)

    # Calculate inertia and silhouette score for the K-means model
    #inertias.append(kmeans.inertia_)
    #score = silhouette_score(X_pca, kmeans.labels_)
    #silhouette_scores.append(score)


# In[6]:


# Choose the optimal K value based on the silhouette scores and elbow curve
#optimal_K = 8

# Perform KMeans clustering with the optimal K value
#kmeans = KMeans(n_clusters=optimal_K, random_state=0)
#kmeans.fit(X)

# Get the cluster labels for each data point
#labels = kmeans.labels_


# In[32]:



# Load the product images as a pandas dataframe, skip the first row
df = pd.read_csv('product_images.csv', header=None, skiprows=1)

# Extract the pixel values of all images as a numpy array
X = df.values.astype('float32')

# Normalize the pixel values
X_norm = X / 255.0


# Determine the optimal number of clusters using the elbow method
n_clusters = 8  # Change this to the optimal number of clusters you obtained

# Apply K-means algorithm to the normalized pixel values with the optimal number of clusters
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X_norm)

# Extract the cluster assignments for each image
cluster_assignments = kmeans.labels_

# Create a new dataframe with the cluster assignments for each image
df_clusters = pd.DataFrame({'image_id': np.arange(len(df)), 'cluster_id': cluster_assignments})


# Define a function to find similar products based on the cluster of the input product
def find_similar_products(product_id, df_clusters, kmeans):
    # Get the cluster assignment for the input product
    cluster_id = df_clusters.loc[product_id, 'cluster_id']
    
    # Find all products in the same cluster
    cluster_products = df_clusters.loc[df_clusters['cluster_id'] == cluster_id, 'image_id'].tolist()
    
    # Remove the input product from the list of similar products
    cluster_products.remove(product_id)
    
    # Calculate the pairwise distances between the input product and all other products in the same cluster
    dist = np.linalg.norm(X_norm[product_id] - X_norm[cluster_products], axis=1)
    
    # Find the indices of the most similar products
    indices = np.argsort(dist)[:10]
    
    # Return the image IDs of the most similar products
    return [cluster_products[i] for i in indices]


# Define the Streamlit app
def app():
    
    # Display an image from a local file
    image = Image.open('./logo.jpg')
    st.image(image, caption="Est.2023", use_column_width=True)
    
    # Centered text using markdown method
    st.markdown("<h1 style='text-align: center;'>Welcome to Shop a lot!</hl>", unsafe_allow_html=True)
    
    # Display instructions to user
    st.write("Enter a product ID or a keyword and we'll recommend 10 similar products, then click search button.")





    #option at the side bar
    analysis = st.sidebar.selectbox('Search product by',['Product ID','Keywords'])



# Allow user to choose between inputting a product ID or keywords
#option = st.radio("Select search method:", ('Product ID', 'Keywords'))

    # Product ID search
    if analysis == 'Product ID':
        # Allow user to input a product ID
        product_id = st.number_input("Enter product ID", min_value=0, max_value=len(df) - 1)
        
        if st.button("Search"):
        # Perform the search and display the results
    


            # Create an image from the pixel array and display it
            input_img = Image.fromarray(X[product_id].reshape(28, 28))
            input_img = input_img.convert("RGB")
            st.image(input_img, caption=f"Input product: {product_id}", width=100)

            # Find similar products based on the user input
            similar_products = find_similar_products(product_id, df_clusters, kmeans)


            for product_id in similar_products:
                # Create an image from the pixel array and display it
                img = Image.fromarray(X[product_id].reshape(28, 28))
                img = img.convert("RGB")
                st.image(img, caption=f"Product ID: {product_id}", width=100)

    # Keyword search
    else:
        # Define a dictionary of cluster keywords
        # For this part, we tried to examine the products in each clusters manually. 
        # Key findings: 1. cluster 0 and cluster 2 are both containing pullovers, clarify the reasons in report
                       #2. Cluster 5 are the images with dark color, which includes almost all categories of products (shoes, shirts, bags, pullovers)
                        # please also clarify the reasons

        cluster_keywords = {
            0: ['pullovers'],
            1: ['shoes'],
            2: ['hoodies'], 
            3: ['tote'],
            4: ['trouser'],
            5: ['black dresses'],
            6: ['boots'],
            7: ['shirts']
            }



        # Allow user to input keywords
        keywords = st.text_input("Enter keywords")
        
        if st.button("Search"):
        # Perform the search and display the results
    





            # Clean up and split the keywords
            keywords = keywords.lower().strip().split(',')

            # Find the relevant clusters for the keywords
            relevant_clusters = []
            for cluster_id, words in cluster_keywords.items():
                for keyword in keywords:
                    if keyword.strip() in words:
                        relevant_clusters.append(cluster_id)
                        break

            # Find all products in the relevant clusters
            relevant_products = []
            for cluster_id in relevant_clusters:
                relevant_products += df_clusters.loc[df_clusters['cluster_id'] == cluster_id, 'image_id'].tolist()

            # Remove duplicates and calculate pairwise distances to the input keywords
            relevant_products = list(set(relevant_products))
            dist = np.linalg.norm(X_norm[relevant_products] - kmeans.cluster_centers_[relevant_clusters], axis=1)

            # Find the indices of the most similar products
            indices = np.argsort(dist)[:10]

            # Return the image IDs of the most similar products
            similar_products = [relevant_products[i] for i in indices]

            # Loop over the similar products and display them
            for product_id in similar_products:
                # Create an image from the pixel array and display it
                img = Image.fromarray(X[product_id].reshape(28, 28))
                img = img.convert("RGB")
                st.image(img, caption=f"Product ID: {product_id}", width=100)

if __name__ == '__main__':
    app()


# In[8]:


dbi = davies_bouldin_score(X_pca, kmeans.labels_)
print('Davies-Bouldin Index: ', dbi)


# In[9]:


# Calculate silhouette score
silhouette = silhouette_score(X_pca, cluster_assignments)
print("Silhouette Score:", silhouette)

# Calculate Calinski-Harabasz Index
ch_score = calinski_harabasz_score(X_pca, cluster_assignments)
print("Calinski-Harabasz Index:", ch_score)


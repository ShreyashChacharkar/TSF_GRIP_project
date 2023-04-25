#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import liberies
import pandas as pd
# liberies for visulalisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy.stats.mstats import trimmed_var
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# # Prepare Data

# ### Import
# 

# In[2]:


df = pd.read_csv("Iris.csv", index_col="Id" )
df.columns = ["Sepal Length(in cm)", "Sepal Width(in cm)","Petal length(in cm)","Petal Width(in cm)", "Species"]
print(df.shape)
df.head()


# ### Explore

# In[3]:


df.isnull().sum()
#no null values


# In[4]:


#unique values in Species
df["Species"].unique()


# In[5]:


cols = df.columns.to_list()
cols


# In[6]:


corr = df[cols[:-1]].corr()
corr.style.background_gradient(axis=None)


# After examining the correlation matrix, it has been determined that Petal length(in cm) and Petal Width(in cm) are strongly correlated (correlation coefficient = 0.96) follwed by Sepal Length(in cm)

# ### Split

# In[7]:


# create feature matrix with mutiple element
X = df[cols[:-1]]
X.head()


# # Build Model

# ### Iterate

# In this project, we are using a pipeline to standardize our features and perform clustering using KMeans. The pipeline consists of two steps: StandardScaler and KMeans.
# 
# The StandardScaler step standardizes our features by subtracting the mean and dividing by the standard deviation. This helps to ensure that all of our features are on the same scale, which can be important for clustering algorithms like KMeans.
# 
# The KMeans step performs clustering using the KMeans algorithm with a specified number of clusters.
# 
# By combining these two steps into a pipeline, we can easily apply the same transformations to our training and test data. This helps to ensure that our model is not overfitting to the training data, and that it will perform well on new data.
# 
# Overall, this pipeline model using StandardScaler and KMeans is an effective way to preprocess our data and perform clustering in a streamlined and efficient manner.
# 

# In[8]:


n_clusters = range(2,13)
inertia_errors = []
silhouette_scores =[]
X_final = X
# Add `for` loop to train model and calculate inertia, silhouette score.
for k in n_clusters:
    model = make_pipeline(StandardScaler(),KMeans(n_clusters=k, random_state=42))
    model.fit(X_final)
    inertia_errors.append(model.named_steps["kmeans"].inertia_)
    silhouette_scores.append(
        silhouette_score(X_final, model.named_steps["kmeans"].labels_))
print("Inertia:", inertia_errors[:3])
print()
print("Silhouette Scores:", silhouette_scores[:3])


# In[9]:


# Plot `Inertia` vs `n_clusters`
plt.plot(n_clusters, inertia_errors)
plt.xlim(1,13)
plt.xlabel("No. of Clusters")
plt.ylabel("Inertia")
plt.title("K-Means Model: Inertia vs Number of Clusters")


# In[10]:


# Plot `silhouette_scores` vs `n_clusters`
plt.plot( n_clusters, silhouette_scores)
plt.xlabel("No. of Clusters")
plt.ylabel("Silhouette Score")
plt.title("K-Means Model: Silhouette Score vs Number of Clusters")


# By analysing both graph we can say that no. of cluster would be 3

# In[11]:


# Build model with best results
final_model = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=42))

# Fit model to data
final_model.fit(X)


# ### Communicating Results

# In[35]:


labels = final_model.named_steps["kmeans"].labels_
xgb = X.groupby(labels).mean()
# Create side-by-side bar chart of `xgb`
fig = px.bar(
    xgb, 
    barmode="group",
    title="Mean Size by Cluster"
)
fig.update_layout(xaxis_title="Cluster", yaxis_title="Size [cm]")
fig.show()


# In[13]:



xgb = X.groupby(df["Species"]).mean()
# Create side-by-side bar chart of `xgb`
fig = px.bar(
    xgb, 
    barmode="group",
    title="Mean Size by Species"
)
fig.update_layout(xaxis_title="Species", yaxis_title="Size [cm]")
fig.show()


# From both the above graph it is clear that Species can be dishtingusih by size. Each cluster represent Species like cluster 2 represents 'Iris-setosa', cluster 0 represents 'Iris-versicolor'and cluster 1 represents  'Iris-virginica. 
# 
# We can observe a similar appereance in bar graph of Iris-versicolor and Iris-virginicasize i.e they have petal length greater than Sepal width Opposite of that Iris-setosa have petal length less than Sepal width

# In[ ]:





# In[37]:


pca = PCA(n_components=2, random_state=42)

# Transform `X`
X_t = pca.fit_transform(X)

# Put `X_t` into DataFrame
X_pca = pd.DataFrame(X_t, columns=["PC1","PC2"])

print("X_pca shape:", X_pca.shape)
X_pca.head()

final_model.fit(X_pca)


# In[50]:


# Create scatter plot of `PC2` vs `PC1`
labels = final_model.named_steps["kmeans"].labels_
centroids = final_model.named_steps["kmeans"].cluster_centers_
df_centroids = pd.DataFrame(centroids, columns= X_pca.columns)
df_centroids
fig = px.scatter(
    data_frame=X_pca,
    x="PC1",
    y="PC2",
    color=labels.astype(str),
    title="PCA Representation of Clusters"
)
fig.add_scatter(
    x=df_centroids['PC1'], y=df_centroids['PC2'],
    mode='markers', marker=dict(size=5, color='black', line=dict(width=2)),
                name='Cluster centroids')
fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
fig.show()


# In[ ]:





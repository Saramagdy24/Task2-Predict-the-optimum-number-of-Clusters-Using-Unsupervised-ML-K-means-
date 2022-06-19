#!/usr/bin/env python
# coding: utf-8

# ### Sara Magdy Mohamed 
# 

# ### Task # 2: Predict the optimum number of cluster using Unsupervised machine learning ( K-means) and represent it visually

# In[47]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk 
from sklearn import datasets
from sklearn import cluster
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[48]:


# import the iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names) 
iris_df


# ### Data Exploration 

# In[49]:


iris_df.shape


# In[50]:


iris_df.head()


# In[51]:


iris_df.info()


# In[52]:


iris_df.describe()


# In[53]:


iris_df.isnull().sum()


# ### Data Visualization

# In[54]:


plt.scatter(iris_df['sepal length (cm)'],iris_df['sepal width (cm)'],c=iris.target, cmap='coolwarm')
plt.xlabel('sepal length (cm)', fontsize=10)
plt.ylabel('sepal width (cm)', fontsize=10)
plt.show()


# In[55]:


plt.scatter(iris_df['petal length (cm)'],iris_df['petal width (cm)'],c=iris.target, cmap='rainbow')
plt.xlabel('petal length (cm)', fontsize=10)
plt.ylabel('petal width (cm)', fontsize=10)
plt.show()


# In[56]:


iris.target


#  ### the Optimum Number of Cluster for  K-means classification 

# In[57]:


wcss = []
no_clusters= range(1,11)
for i in no_clusters:
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(iris_df)
    wcss.append(kmeans.inertia_)


# In[58]:


# Plotting the results onto a line graph, 
plt.plot(no_clusters, wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# ### Apply K-means to dataset  / Creating the kmeans classifier

# In[59]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(iris_df.values)
y_kmeans


# ### Cluster Visualization

# In[62]:


# Visualising the clusters - On the first two columns (spetal length ,spetal width)
plt.figure(figsize=(10,8))
# 0,0 means first cluster and first column , 0,1 means first cluster and second column
plt.scatter(iris_df.values[y_kmeans==0,0],iris_df.values[y_kmeans==0,1],s=75,label='Iris-setosa')
# 1,0 means first cluster and first column , 1,1 means first cluster and second column
plt.scatter(iris_df.values[y_kmeans==1,0],iris_df.values[y_kmeans==1,1],s=75,label='Iris-versicolour')
# 2,0 means first cluster and first column , 2,1 means first cluster and second column
plt.scatter(iris_df.values[y_kmeans==2,0],iris_df.values[y_kmeans==2,1],s=75,label='Iris-virginica')

# plotting centroids of clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],s = 75, label = 'Centroids' )

plt.legend()
plt.show()


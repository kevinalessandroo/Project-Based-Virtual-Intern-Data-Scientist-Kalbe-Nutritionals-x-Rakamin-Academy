#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[4]:


df = pd.read_csv("C:\Kuliah\INTERNSHIPS\Kalbe Nutritionals Internship\Final Project\Data\Merged_Data.csv")
df


# Data yang digunakan sama dengan data pada analisis time series forecasting, sehingga sudah tidak perlu dilakukan data preprocessing karena sudah dilakukan pada analisis time series forecasting.

# # Clustering Analysis (Machine Learning)

# In[5]:


df.head()


# Membuat data baru untuk clustering, yaitu groupby by customerID lalu yang di aggregasi adalah : </br>
# ○ Transaction id count </br>
# ○ Qty sum </br>
# ○ Total amount sum

# In[6]:


df_clust = df.groupby('CustomerID').aggregate({'TransactionID':'count','Qty':'sum','TotalAmount':'sum'})
df_clust


# Menggunakan metode clustering KMeans

# In[10]:


scaler = StandardScaler()
scaler.fit(df_clust)
df_scaled = scaler.transform(df_clust)
df_scaled


# In[11]:


df_scaled = pd.DataFrame(df_scaled, columns=['Transaction ID','Qty','Total Amount'])
df_scaled


# In[12]:


kmeans = KMeans(n_clusters=4, max_iter=500)
kmeans.fit(df_scaled)


# In[14]:


kmeans.labels_


# # Elbow Curve

# In[15]:


# Elbow-curve/SSD
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=500)
    kmeans.fit(df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(ssd)


# Pada elbow curve method, lokasi ‘tikungan’ yang terbentuk di plot, pada umumnya dianggap sebagai indikator jumlah cluster yang tepat. Akan tetapi nilai k ‘optimal’ yang diperoleh dari metode elbow curve, sering kali bersifat “ambigu” atau belum pasti akan menghasilkan jumlah cluster (k) yang optimal. Pada grafik di atas, saya menentukan k optimal =3 atau 4 karena grafik yang dihasilkan membentuk tikungan dan setelahnya mulai melandai. Namun, untuk lebih memastikan bahwa k yang dipilih optimal, akan dilakukan analisis dengan menggunakan silhouette analysis.

# # Silhouette Analysis Method

# In[17]:


# Silhouette Analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    
    # Initialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=500)
    kmeans.fit(df_scaled)
    cluster_labels = kmeans.labels_
    
    # Silhouette Score
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# For n_clusters=2, the silhouette score is 0.48530485981509097 </br>
# For n_clusters=3, the silhouette score is 0.4286478086768933 </br>
# For n_clusters=4, the silhouette score is 0.3772419295085858 </br>
# For n_clusters=5, the silhouette score is 0.34207742281768294 </br>
# For n_clusters=6, the silhouette score is 0.30030062675665914 </br>
# For n_clusters=7, the silhouette score is 0.27812711646035787 </br>
# For n_clusters=8, the silhouette score is 0.2911924648482836 </br>
# 
# Berdasarkan output tersebut, dapat disimpulkan bahwa untuk n_clusters = 2 menghasilkan nilai silhouette yang tertinggi. Namun, pada analisis ini akan tetap digunakan n_clusters = 3 untuk melihat segmentasi yang lebih variatif

# In[25]:


# Final model with k=3
kmeans = KMeans(n_clusters = 3, max_iter = 500)
kmeans.fit(df_scaled)


# In[26]:


# Assign the label
df_clust['Cluster_Id'] = kmeans.labels_
df_clust.head(100)


# In[27]:


df_clust.groupby(['Cluster_Id']).agg({
    'TransactionID' : 'mean',
    'Qty' : 'mean',
    'TotalAmount' : 'mean',
})


# In[29]:


# Select two features for visualization (change these to the desired features)
x_feature = 'Qty'
y_feature = 'TotalAmount'

# Scatter plot
plt.figure(figsize=(10, 6))
for Cluster_Id, cluster_data in df_clust.groupby('Cluster_Id'):
    plt.scatter(cluster_data[x_feature], cluster_data[y_feature], label=f'Cluster {Cluster_Id}')

plt.title('Clustering Visualization')
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





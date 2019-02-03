#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import glob
import json
import os, sys
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pathin="/Volumes/lia_595gb/randel/python/dados/recorte/"
pathout="/Volumes/lia_595gb/randel/python/dados/recorte/figures/"


# In[3]:


pathin


# In[4]:


for file in os.listdir(pathin):
    if file.endswith(".csv"):
        name = os.path.splitext(file)[0]
        df = pd.read_csv(os.path.join(pathin, file), sep=',', decimal='.')


# In[5]:


df.head()


# In[6]:


cols=df.columns


# In[7]:


cols


# In[8]:


tch=cols[8:21]
rr=cols[6]
lch=cols[8:15]
hch=cols[15:21]


# In[9]:


tch


# In[10]:


TB=df[tch]


# In[12]:


TB.head()


# In[11]:


df2= df[tch].copy() 


# In[12]:


df2.head()


# In[13]:


# How to insert new column in the DataFrame:

idx = 0
new_col = df[rr] # can be a list, a Series, an array or a scalar   
df2.insert(loc=idx, column='rr', value=new_col)


# In[14]:


df2.head()


# In[15]:


from sklearn.decomposition import PCA


# In[16]:


# Choosing the number of components:

pca_ver = PCA().fit(TB)
plt.plot(np.cumsum(pca_ver.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');


# In[17]:


# Comparing the data before and after the transformation in components:

pca = PCA(n_components=2)
pca.fit(TB)
TB_transformed = pca.transform(TB)
print("original shape:   ", TB.shape)
print("transformed shape:", TB_transformed.shape)


# In[27]:


#  Plot the first two principal components of each point to learn about the data:
plt.figure(figsize=(12,8))
plt.scatter(TB_transformed[:, 0], TB_transformed[:, 1],
            c=df[rr], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('YlGn', 10))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar();


# In[18]:


pca.components_


# In[19]:


pca.explained_variance_


# In[21]:


pca.explained_variance_ratio_


# In[22]:


pca.singular_values_


# In[23]:


pca.mean_


# In[24]:


pca.noise_variance_


# In[34]:


pca.components_.shape


# In[37]:


tch


# In[38]:


df_comp1=pd.DataFrame(pca.components_, columns=tch)


# In[39]:


df_comp1


# In[41]:


plt.figure(figsize=(12,8))
sns.heatmap(df_comp1, cmap='plasma')


# In[45]:


TB_transformed.shape


# In[46]:


print('NumPy covariance matrix: \n%s' %np.cov(TB_transformed.T))


# In[65]:


# Plot of correlation between the first component and the rain:
plt.figure(figsize=(12,8))
plt.scatter(df[rr], TB_transformed[:, 0],edgecolor='none', alpha=0.5,
            cmap= 'blue')
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 1')


# In[64]:


# Plot of correlation between the first component and the rain:
plt.figure(figsize=(12,8))
plt.scatter(df[rr], TB_transformed[:, 1], c='red', edgecolor='none', alpha=0.5)
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 2')


# In[ ]:





# In[67]:


from sklearn.preprocessing import StandardScaler


# In[68]:


scaler = StandardScaler()


# In[69]:


scaler.fit(TB)


# In[70]:


scaled_TB = scaler.transform(TB)


# In[71]:


scaled_TB


# In[72]:


pca.fit(scaled_TB)


# In[74]:


TB_sc_transformed=pca.transform(scaled_TB)


# In[75]:


TB_sc_transformed.shape


# In[76]:


# Plot of correlation between the first component and the rain:
plt.figure(figsize=(12,8))
plt.scatter(df[rr], TB_sc_transformed[:, 0],edgecolor='none', alpha=0.5,
            cmap= 'green')
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 1')


# In[77]:


# Plot of correlation between the first component and the rain:
plt.figure(figsize=(12,8))
plt.scatter(df[rr], TB_sc_transformed[:, 1], c='red', edgecolor='none', alpha=0.5)
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 2')


# In[29]:


#  Plot the first two principal components of each point to learn about the data:
plt.figure(figsize=(12,8))
plt.scatter(TB_sc_transformed[:, 0], TB_sc_transformed[:, 1],
            c=df[rr], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('YlGn', 10))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar();


# In[25]:


pca4 = PCA(n_components=4)
pca4.fit(TB)
TB_transformed4 = pca4.transform(TB)
print("original shape:   ", TB.shape)
print("transformed shape:", TB_transformed4.shape)


# In[47]:


#  Plot the first two principal components of each point to learn about the data:
plt.figure(figsize=(12,8))
plt.scatter(TB_transformed4[:, 0], TB_transformed4[:, 1],
            c=df[rr], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('YlGn', 10))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA n_components = 4')
plt.colorbar();


# In[48]:


pca4.components_


# In[49]:


pca4.explained_variance_


# In[50]:


pca4.explained_variance_ratio_


# In[ ]:





# In[ ]:





# In[ ]:





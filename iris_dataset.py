#!/usr/bin/env python
# coding: utf-8

# In[35]:


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np


# In[36]:


iris=datasets.load_iris()
print(iris.data)


# In[37]:


print(iris.target)


# In[38]:


x=pd.DataFrame(iris.data)
x.columns=['sepal_length' , 'sepal_width', 'petal_length', 'petal_width']
y=pd.DataFrame(iris.target)
y.columns=['targets']
colormap=np.array(['red', 'lime', 'black'])
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.scatter(x.sepal_length, x.sepal_width, c=colormap[y.targets], s=40)
plt.title('Sepal Data')
plt.show()


# In[39]:


plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.scatter(x.petal_length, x.petal_width, c=colormap[y.targets], s=40)
plt.title('Petal Data')
plt.show()


# In[55]:


plt.figure(figsize=(14, 7))
colormap=np.array(['red', 'lime', 'black'])
plt.subplot(1, 2, 1)
plt.scatter(x.petal_length, x.petal_width, c=colormap[y.targets], s=40)
plt.title('Petal Data Real Classification')


model=KMeans(n_clusters=3)
model.fit(x)
print(model.labels_)


plt.subplot(1, 2, 1)
plt.scatter(x.petal_length, x.petal_width, c=colormap[model.labels_], s=40)
plt.title('Petal Data KMeans Classification')


# In[56]:


predY=np.choose(model.labels_, [0, 1, 2]).astype(np.int64)
print(model.labels_)
print(predY)


# In[57]:


print(sm.accuracy_score(y, predY))
print(sm.confusion_matrix(y, predY))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





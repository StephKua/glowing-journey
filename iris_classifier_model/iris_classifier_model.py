#!/usr/bin/env python
# coding: utf-8

# # Tutorial 1

# ## In this exercise, we are comparing 3 different types of classification model on the given dummy dataset.
# 
# 1. SGD
# 2. Linear SVC
# 3. Random Forest

# In[1]:


import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# In[2]:


iris = datasets.load_iris()


# In[3]:


train_df = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
train_df.head()


# In[4]:


test_df = pd.DataFrame(iris.target, columns=['Types'])
test_df['Types'].value_counts()


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(train_df, test_df, test_size=0.2, random_state=20)
y_train = y_train.values.reshape(-1)
y_test = y_test.values.reshape(-1)


# In[6]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[7]:


clf_svc = LinearSVC()
clf_sgd = SGDClassifier()
clf_forest = RandomForestClassifier()


# In[8]:


clf_svc.fit(X_train, y_train)
clf_sgd.fit(X_train, y_train)
clf_forest.fit(X_train, y_train)


# In[9]:


pred = clf_svc.predict(X_test)
svc_score = accuracy_score(y_test, pred)


# In[10]:


pred = clf_sgd.predict(X_test)
sgd_score = accuracy_score(y_test, pred)


# In[11]:


pred = clf_forest.predict(X_test)
forest_score = accuracy_score(y_test, pred)


# In[12]:


print(sgd_score, svc_score, forest_score )


# In[ ]:





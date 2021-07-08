#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas
from pandas import DataFrame
import matplotlib.pyplot as plot

from sklearn.linear_model import LinearRegression


# In[3]:


data = pandas.read_csv('cost_revenue_sheet.csv')
# data.describe()


# In[4]:


X = DataFrame(data, columns=['production_budget'])
y = DataFrame(data, columns=['worldwide_gross'])


# In[5]:


# matplotlib.pyplot.scatter(X,y)
# matplotlib.pyplot.show()
plot.figure(figsize=(10,6))
plot.title('Film Cost vs Global Revenue')
plot.xlabel('Production Cost $')
plot.ylabel('Worldwide Revenue')
plot.ylim(0,3000000000) 
plot.xlim(0,500000000)
plot.scatter(X,y,alpha = 0.2)
plot.show()


# In[6]:


regression = LinearRegression()
regression.fit(X,y)


# In[7]:


regression.coef_ # this is the slope


# In[8]:


regression.intercept_ # this is the y intercept


# In[9]:


plot.figure(figsize=(10,6))
plot.title('Film Cost vs Global Revenue')
plot.xlabel('Production Cost $')
plot.ylabel('Worldwide Revenue')
plot.ylim(0,3000000000) 
plot.xlim(0,500000000)
plot.scatter(X,y,alpha = 0.2)
plot.plot(X, regression.predict(X), color='red', linewidth= 3)
plot.show()


# In[10]:


regression.score(X,y)


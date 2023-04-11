#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv("http://bit.ly/w-data")


# In[3]:


df.head()


# In[4]:


#for checking null values
df.info()


# In[5]:


model = LinearRegression()


# In[6]:


X = df.drop("Scores", axis =1)
Y = df["Scores"]
model.fit(X,Y)


# In[ ]:





# In[23]:


#prediction of model for 9.25 hrs/day
new_input = [[9.25]]
req_output = model.predict(new_input)
print(f"from the model if student study  for {new_input[0][0]} hrs/day the score will be {req_output[0]}")


# In[26]:


plt.scatter(X.values,Y.values)
plt.plot(X.values, model.predict(X), c="r")
plt.scatter(new_input,model.predict(new_input), c="k")
plt.xlabel("Hours Per Day")
plt.ylabel("Scores")
plt.title("Linear Model of Studying hour and scores of students")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  


# In[3]:


df = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
print("Data imported successfully")


# In[4]:


df.head(10)


# In[5]:


df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[6]:


X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values 


# In[7]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[8]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# In[12]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[10]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[13]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# In[19]:


# You can also test with your own data
hours = 9.25
predicted_score = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(predicted_score[0]))


# In[20]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:





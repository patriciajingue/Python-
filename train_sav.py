#!/usr/bin/env python
# coding: utf-8

# In[39]:


cd Downloads


# In[40]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# In[41]:


df= pd.read_csv('gesta.csv')
df


# In[38]:


df


# In[22]:


df.PGMSUGUR.unique()


# In[43]:


df['BORNCONGEN'] = df['BORNCONGEN'].map({10: 1,21:2,23:2,22:1})
df['PGMTRANQFREQ'] = df['PGMTRANQFREQ'].map({10: 1,26:2,21:2,23:1,24:2,22:2,25:2})
df['PGMECLAMP'] = df['PGMECLAMP'].map({10: 1,11:2,13:2,12:1})
df['PGMSUGUR'] = df['PGMSUGUR'].map({2: 1,3:2,})





# In[44]:


df.isnull().sum()


# In[45]:


x= df.iloc[:,:-1]
print(x.shape)
y = df.iloc[:,-1:]
print(y.shape)


# In[46]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size= 0.1, random_state=10)


# In[47]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# In[48]:


model = LogisticRegression()
grid={'C':10.0**np.arange(-2,3), 'penalty':['l''l2']}
cv=KFold(n_splits=5, shuffle=False)


# In[49]:


clf= GridSearchCV(model,grid, cv=cv, n_jobs=-1, scoring='f1_macro')
clf.fit(x_train,y_train)


# In[50]:


from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot
from numpy import where
from sklearn.metrics import plot_confusion_matrix


# In[51]:


from imblearn.under_sampling import RandomUnderSampler
rus= RandomUnderSampler(random_state=1)
x_train_resampled,y_train_resampled = rus.fit_resample(x_train,y_train)

unique_elements, counts_elements= np.unique(y_train_resampled, return_counts=True)
print(unique_elements, counts_elements)


# In[52]:


import seaborn as sns


# In[53]:


clf= LogisticRegression().fit(x_train_resampled,y_train_resampled)
y_pred= clf.predict(x_test)
sns.set_context('poster')
disp= plot_confusion_matrix(clf, x_test, y_test,
                           cmap= 'cividis', colorbar=False)


# In[54]:


print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))


# In[55]:


y_pred_proba= clf.predict_proba(x_test)[:,1]
y_pred_proba


# In[56]:


plt.subplots(figsize = (10,5))
sns.regplot(y_test, y_pred)


# In[57]:


import pickle
filename='train.sav'
pickle.dump(clf,open(filename,'wb'))


# In[58]:


loaded_model= pickle.load(open('train.sav','rb'))


# In[59]:


sns.distplot(y_pred)


# In[21]:


input_data= (1,10,0,0,1,2,0,0,0,0,1,10,1,1,1,26)
#change to numpy array
input_numpy = np.asarray(input_data)
input_data_reshape= input_numpy.reshape(1,-1)

prediction = loaded_model.predict_proba(input_data_reshape)
print(prediction)


# In[ ]:





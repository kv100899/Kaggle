#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# In[2]:


data = pd.read_csv('train (1).csv')
data


# In[3]:


data.isnull().sum().sort_values(ascending=False)


# In[4]:


#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[5]:


dataset = data.copy()


# In[6]:


dataset = dataset.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=1)


# In[7]:


dataset.isnull().sum()


# In[8]:


data_no_mv = dataset.dropna(axis=0)


# In[9]:


#correlation matrix
corrmat = data_no_mv.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# In[10]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data_no_mv[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[11]:


sns.distplot(data_no_mv['SalePrice'])


# In[12]:


q = data_no_mv['SalePrice'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['SalePrice']<q]


# In[13]:


sns.distplot(data_1['SalePrice'])


# In[14]:


sns.distplot(data_1['OverallQual'])


# In[15]:


sns.distplot(data_1['GrLivArea'])


# In[16]:


s = data_1['GrLivArea'].quantile(0.99)
data_2 = data_1[data_1['GrLivArea']<s]


# In[17]:


sns.distplot(data_2['GrLivArea'])


# In[18]:


sns.distplot(data_2['GarageCars'])


# In[19]:


a = data_2['GarageCars'].quantile(0.99)
data_3 = data_2[data_2['GarageCars']<a]


# In[20]:


sns.distplot(data_3['GarageCars'])


# In[21]:


sns.distplot(data_3['TotalBsmtSF'])


# In[22]:


sns.distplot(data_3['FullBath'])


# In[23]:


sns.distplot(data_3['TotRmsAbvGrd'])


# In[24]:


sns.distplot(data_3['YearBuilt'])


# In[25]:


v = data_3['YearBuilt'].quantile(0.01)
data_4 = data_3[data_3['YearBuilt']>v]


# In[26]:


sns.distplot(data_4['YearBuilt'])


# In[27]:


data_cleaned = data_4.reset_index(drop=True)


# In[28]:


log_price = np.log(data_cleaned['SalePrice'])

# Then we add it to our data frame
data_cleaned['log_price'] = log_price
data_cleaned


# In[29]:


data_cleaned = data_cleaned.drop(['SalePrice'],axis=1)


# In[30]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# To make this as easy as possible to use, we declare a variable where we put
# all features where we want to check for multicollinearity
# since our categorical data is not yet preprocessed, we will only take the numerical ones
variables = data_cleaned[['OverallQual','GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath','TotRmsAbvGrd' ]]

# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = variables.columns


# In[31]:


vif


# In[32]:


data_cleaned.columns.values


# In[33]:


cols = ['OverallQual','GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath','TotRmsAbvGrd' , 'YearBuilt', 'log_price']


# In[34]:


data_cleaned = data_cleaned[cols]


# In[35]:


data_cleaned


# In[36]:


data_with_dummies = pd.get_dummies(data_cleaned, drop_first=True)


# In[37]:


data_with_dummies


# In[38]:


X = data_with_dummies.iloc[:, :-1].values
y = data_with_dummies.iloc[:, -1].values


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[ ]:





# In[40]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[41]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[42]:


df_pf = pd.DataFrame(np.exp(y_pred), columns=['Prediction'])
df_pf.head()


# In[43]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[44]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:





# In[ ]:





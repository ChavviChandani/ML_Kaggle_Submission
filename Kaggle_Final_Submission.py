#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary Libraries

# In[89]:


import pandas as pd
import numpy as np


# ### Importing Training and Testing Data

# In[90]:


df_train=pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
df_test=pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')


# ### To view first 5 rows of training data set

# In[91]:


df_train.head()


# ### To view last 5 rows of testing data set

# In[92]:


df_test.tail()


# ### To view basic information about the Training data set. eg. Missing values, datatype,etc

# In[93]:


df_train.info()


# ### To get a better understanding of the Training data

# In[94]:


df_train.describe()


# ### Finding the count of NAN value records in Training Data set

# In[95]:


print("No. of missing values in Gender:",df_train.Gender.isna().sum())
print("No. of missing values in Year of Record:",df_train['Year of Record'].isna().sum())
print("No. of missing values in Age:",df_train.Age.isna().sum())
print("No. of missing values in Profession:",df_train.Profession.isna().sum())
print("No. of missing values in University Degree:",df_train['University Degree'].isna().sum())
print("No. of missing values in Hair Color:",df_train['Hair Color'].isna().sum())


# ### Handling missing values in Gender

# In[96]:


#Finding unique values for Gender Column
print(df_train.Gender.unique())


# In[97]:


#Replacing NAN with unknown
df_train['Gender']=df_train['Gender'].fillna('Unknown')

#Replacing 0 with unknown
df_train['Gender']=df_train['Gender'].replace('0','Unknown')

#Replacing unknown with Unknown
df_train['Gender']=df_train['Gender'].replace('unknown','Unknown')


# In[98]:


#Finding the count of NAN value records in the Training Data Set for Gender
df_train['Gender'].isna().sum()


# In[99]:


#Finding unique values for Gender Column
df_train.Gender.unique()


# ### Handling Missing values in Year of Record

# In[100]:


#Finding unique values for Year of Record
list(df_train['Year of Record'].unique())


# In[101]:


#Replacing NAN with unknown
df_train['Year of Record']=df_train['Year of Record'].fillna(0)


# In[102]:


#To check if the Nan values were removed from Year of Record
df_train.info()


# ### Handling Missing values in Age

# In[103]:


#Finding unique values for Age
list(df_train['Age'].unique())


# In[104]:


#Replacing NAN with 0
df_train['Age']=df_train['Age'].fillna(0)


# In[105]:


#Finding if any NAN values are present in Age
df_train['Age'].isna().sum()


# ### Handling missing values in Profession

# In[106]:


#Finding unique values for Age
list(df_train['Profession'].unique())


# In[107]:


#Replacing NAN with unknown
df_train['Profession']=df_train['Profession'].fillna('Unknown')


# ### Handling missing values in University Degree

# In[108]:


#Finding unique values in University Degree
df_train['University Degree'].unique()


# In[109]:


#Replacing NAN with unknown
df_train['University Degree']=df_train['University Degree'].fillna('Unknown')

#Replacing 0 with unknown
df_train['University Degree']=df_train['University Degree'].replace('0','Unknown')


# ### Handling missing values in Hair Color

# In[110]:


#Finding unique values in Hair Color
df_train['Hair Color'].unique()


# In[111]:


#Replacing NAN with unknown
df_train['Hair Color']=df_train['Hair Color'].fillna('Unknown')

#Replacing 0 with unknown
df_train['Hair Color']=df_train['Hair Color'].replace('0','Unknown')

#Replacing unknown with Unknown
df_train['Hair Color']=df_train['Hair Color'].replace('unknown','Unknown')


# ### To check if all the missing values were removed from Training Dataset

# In[112]:


df_train.info()


# ### Handling missing data from Testing Data Set

# In[113]:


#Replacing NAN with unknown
df_test['Gender']=df_test['Gender'].fillna('Unknown')

#Replacing 0 with unknown
df_test['Gender']=df_test['Gender'].replace('0','Unknown')

#Replacing unknown with Unknown
df_test['Gender']=df_test['Gender'].replace('unknown','Unknown')

#Replacing NAN with unknown
df_test['Year of Record']=df_test['Year of Record'].fillna(0)

#Replacing NAN with 0
df_test['Age']=df_test['Age'].fillna(0)

#Replacing NAN with unknown
df_test['Profession']=df_test['Profession'].fillna('Unknown')

#Replacing NAN with unknown
df_test['University Degree']=df_test['University Degree'].fillna('Unknown')

#Replacing 0 with unknown
df_test['University Degree']=df_test['University Degree'].replace('0','Unknown')

#Replacing NAN with unknown
df_test['Hair Color']=df_test['Hair Color'].fillna('Unknown')

#Replacing 0 with unknown
df_test['Hair Color']=df_test['Hair Color'].replace('0','Unknown')

#Replacing unknown with Unknown
df_test['Hair Color']=df_test['Hair Color'].replace('unknown','Unknown')


# ### Implementing Target Encoding Function

# In[114]:


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(df_train, df_test,target):
    min_samples_leaf=1
    smoothing=1
    noise_level=0
    temp = pd.concat([df_train, target], axis=1)
    #Compute target mean 
    averages = temp.groupby(by=df_train.name)[target.name].agg(["mean", "count"])
    #Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
 
    #Apply average function to all target data
    prior = target.mean()
    #The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    #Apply averages to trn and tst series
    ft_df_train = pd.merge(df_train.to_frame(df_train.name),averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),on=df_train.name,how='left')['average'].rename(df_train.name + '_mean').fillna(prior)
    #pd.merge does not keep the index so restore it
    ft_df_train.index = df_train.index 
    ft_df_test = pd.merge(df_test.to_frame(df_test.name),averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),on=df_test.name,how='left')['average'].rename(df_train.name + '_mean').fillna(prior)
    #pd.merge does not keep the index so restore it
    ft_df_test.index = df_test.index
    return add_noise(ft_df_train, noise_level), add_noise(ft_df_test, noise_level)


# ### Implementing Target Encoder on Categorical Columns

# In[115]:


df_train['Gender'],df_test['Gender']=target_encode(df_train['Gender'], df_test['Gender'],df_train['Income in EUR'])
df_train['Country'],df_test['Country']=target_encode(df_train['Country'], df_test['Country'],df_train['Income in EUR'])
df_train['Profession'],df_test['Profession']=target_encode(df_train['Profession'], df_test['Profession'],df_train['Income in EUR'])
df_train['University Degree'],df_test['University Degree']=target_encode(df_train['University Degree'], df_test['University Degree'],df_train['Income in EUR'])
df_train['Hair Color'],df_test['Hair Color']=target_encode(df_train['Hair Color'], df_test['Hair Color'],df_train['Income in EUR'])


# ### Dividing the Training Data to X= Features and y=Label for Testing Accuracy

# In[116]:


X=df_train.drop(['Income in EUR'],axis=1)
y=df_train['Income in EUR']


# ### Importing library from sklearn to Split Main Training Data to Train and Test

# In[117]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7, random_state=100)


# ### Implementing Ordinary Least Squares for feature reduction depending upon P value
# #### Remove the columns if P value is high as they as insignificant

# In[118]:


import statsmodels.api as sm

X_train1 = sm.add_constant(X_train)

lm = sm.OLS(y_train,X_train1).fit()

print(lm.summary())


# ### Dropping insignificant columns from Test and Train

# In[119]:


X_train=X_train.drop(['Hair Color'],axis=1)
X_train=X_train.drop(['Gender'],axis=1)
X_train=X_train.drop(['Wears Glasses'],axis=1)
X_train=X_train.drop(['Instance'],axis=1)


X_test=X_test.drop(['Hair Color'],axis=1)
X_test=X_test.drop(['Gender'],axis=1)
X_test=X_test.drop(['Wears Glasses'],axis=1)
X_test=X_test.drop(['Instance'],axis=1)


# ### Implementing GradientBoostingRegressor 

# In[120]:


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=1200,learning_rate=0.09)
gbr.fit(X_train,y_train)


# In[121]:


X_train


# ### Predicting y_pred from X_test

# In[122]:


y_pred = gbr.predict(X_test)


# ### Calculating Root Mean Square Error

# In[123]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred) 
rmse = np.sqrt(mse)

print ('{0:f}'.format(rmse))


# ### Splitting the Main Testing Data to features and label

# In[124]:


X_test_final=df_test.drop(['Income'],axis=1)
y_test_final=df_test['Income']


# ### Dropping all the insignificant columns as we did for training data

# In[125]:


X_test_final=X_test_final.drop(['Hair Color'],axis=1)
X_test_final=X_test_final.drop(['Gender'],axis=1)
X_test_final=X_test_final.drop(['Wears Glasses'],axis=1)
X_test_final=X_test_final.drop(['Instance'],axis=1)


# ### Predicting the final Income

# In[126]:


y_pred_final= gbr.predict(X_test_final)


# ### Preparing the submission file columns

# In[127]:


Income=pd.DataFrame(y_pred_final)
Instance=pd.DataFrame(df_test['Instance'])
submission_df=pd.concat([Instance,Income],axis=1)
submission_df.columns=['Instance','Income']
submission_df


# In[129]:


submission_df.to_csv('Final_Submission.csv',index=False)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # The Best Classifier
# 

# In[6]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# # About Database
# This dataset is about past loans. The Loan_train.csv data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:

# | Feild | Description |
# | :- | :- |
# |Loan_status | : Whether a loan is paid off on in collection |
# | Principal | : Basic principal loan amount at the |
# | Terms | : Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | : When the loan got originated and took effects |
# | Due_date | : Since it’s one-time payoff schedule, each loan has one single due date |
# | Age | : Age of applicant |
# | Education | : Education of applicant |
# | Gender | : The gender of applicant |

# Now we'll download the dataset

# In[2]:


get_ipython().system('wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv')


# # Load Data From CSV File

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# # Convert to date time object
# 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# Let’s see how many of each class is in our data set
# 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection
# 
# *Let's plot some columns to underestand data better:*

# In[7]:


get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing: Feature selection/extraction
# Let's look at the day of the week people get the loan
# 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# # Convert Categorical features to numerical values
# 
# Let's look at gender:

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 
# Let's convert male to 0 and female to 1:

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# # One Hot Encoding
# 
# **How about education?**

# In[14]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# **Features before One Hot Encoding**

# In[15]:


df[['Principal','terms','age','Gender','education']].head()


# **Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame**

# In[16]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# # Feature Selection
# 
# Let's define feature sets, X:

# In[17]:


X = Feature
X[0:5]


# What are our lables?

# In[18]:


y = df['loan_status'].values
y[0:5]


# # Normalize Data
# Data Standardization give data zero mean and unit variance (technically should be done after train test split)

# In[19]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification
# 
# Now, use the training set to build an accurate model. Then use the test set to report the accuracy of the model.
# We should use the following algorithm:
# 
# * K Nearest Neighbor(KNN)
# * Decision Tree
# * Support Vector Machine
# * Logistic Regression
# 
# __ Notice:__
# 
# * We can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# * We should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# * We should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: We should find the best k to build the model with the best accuracy.\ **warning**: We should not use the loan_test.csv for finding the best k, however, we can split our train_loan.csv into train and test to find the best k.

# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))


# In[23]:


knn = KNeighborsClassifier(n_neighbors = 4).fit(X_train,y_train)
Pred_y = knn.predict(X_test)
print("Accuracy of model at K=4 is",metrics.accuracy_score(y_test, Pred_y))


# # Decision Tree

# In[24]:


from sklearn.tree import DecisionTreeClassifier


# In[25]:


dtree = DecisionTreeClassifier(criterion="entropy",max_depth=2).fit(X_train,y_train)


# In[26]:


y_pred = dtree.predict(X_test)
print("Accuracy of Decision Tree is",metrics.accuracy_score(y_test, y_pred))


# # Support Vector Machine
# 

# In[27]:


from sklearn.svm import SVC


# In[28]:


svm = SVC().fit(X_train,y_train)


# In[29]:


y_pred = svm.predict(X_test)
print("Accuracy of SVM is",metrics.accuracy_score(y_test, y_pred))


# # Logistic Regression
# 

# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


lr = LogisticRegression().fit(X_train,y_train)


# In[32]:


y_pred = lr.predict(X_test)
print("Accuracy of Logistic Regression is",metrics.accuracy_score(y_test, y_pred))


# # Model Evaluation using Test set
# 

# In[33]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# **First, download and load the test set:**

# In[34]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# **Load Test set for evaluation**

# In[35]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[36]:


test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

Feature = test_df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(test_df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# In[37]:


X= Feature
X= preprocessing.StandardScaler().fit(X).transform(X)
y= test_df['loan_status'].values


# In[39]:


# Final Report
j_scores = [jaccard_score(y,knn.predict(X),pos_label="PAIDOFF"),jaccard_score(y,dtree.predict(X),pos_label="PAIDOFF"),jaccard_score(y,svm.predict(X),pos_label="PAIDOFF"),jaccard_score(y,lr.predict(X),pos_label="PAIDOFF")]

f_scores = [f1_score(y,knn.predict(X),pos_label="PAIDOFF"),f1_score(y,dtree.predict(X),pos_label="PAIDOFF"),f1_score(y,svm.predict(X),pos_label="PAIDOFF"),f1_score(y,lr.predict(X),pos_label="PAIDOFF"),]

lg_loan_status_probas = lr.predict_proba(X)
lg_log_loss = log_loss(y, lg_loan_status_probas)
log_losses = ["NA","NA","NA",lg_log_loss]

report = pd.DataFrame(list(zip(j_scores,f_scores,log_losses)),
                  columns =['Jaccard', 'F1-score','LogLoss'],
                  index=['KNN','Decision Tree','SVM','Logistic Regression'])


report


# **This is the final result shown here**

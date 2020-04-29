#!/usr/bin/env python
# coding: utf-8

# In[104]:


import pandas as pd
import numpy as np


# In[105]:


train = pd.read_csv("Mobile Price Classification/train.csv")


# In[106]:


train


# In[107]:


test = pd.read_csv("Mobile Price Classification/test.csv")
test = test.drop("id" , axis = 1)


# In[108]:


train.columns


# # Variables
# 
# 
# ##### id-ID
# ##### battery_power-Total energy a battery can store in one time measured in mAh
# ##### blue-Has bluetooth or not
# ##### clock_speed-speed at which microprocessor executes instructions
# ##### dual_sim-Has dual sim support or not
# ##### fc-Front Camera mega pixels
# ##### four_g-Has 4G or not
# ##### int_memory-Internal Memory in Gigabytes
# ##### m_dep-Mobile Depth in cm
# ##### mobile_wt-Weight of mobile phone
# ##### n_cores-Number of cores of processor
# ##### pc-Primary Camera mega pixels
# ##### px_height-Pixel Resolution Height
# ##### px_width-Pixel Resolution Width
# ##### ram-Random Access Memory in Megabytes
# ##### sc_h-Screen Height of mobile in cm
# ##### sc_w-Screen Width of mobile in cm
# ##### talk_time-longest time that a single battery charge will last when you are
# ##### three_g-Has 3G or not
# ##### touch_screen-Has touch screen or not
# ##### wifi-Has wifi or not

# In[109]:


# For those whose prince range is not given we will remove the data.
train = train.dropna(subset = ["price_range"])


# In[110]:


train


# ### Now lets check out the null values present in each column

# In[111]:


train.isnull().sum()


# In[112]:


train.dtypes


# In[113]:


# Thankfully there is no null values present in the data frame.


# #### Since we have a lot of features to choose from. We must check how the features relate to each other and how they individually influence the price range.

# In[114]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corr_data = train.corr()
plt.subplots(figsize = (20,15))
sns.heatmap(corr_data, xticklabels=corr_data.columns, yticklabels=corr_data.columns, annot=True, fmt = ".1g",vmin = -1,vmax = 1,center = 0,cmap = "coolwarm")


# ### Major takeaways from the plot would be 
# #### 1. RAM Highly influences the prince range.
# #### 2. Front Camera and Back Camera are also correlated hence one of them could be dropped. 

# # Try 1 - No Variable Dropped. All used in the pipeline.

# In[166]:


X = train[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g','touch_screen', 'wifi']]
Y = train["price_range"]


# #### 1.Logistic Regression

# In[116]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[117]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# #### 2. Support Vector Machine

# In[167]:


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='linear') 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
clf.fit(X_train, y_train)


# In[168]:


y_pred = clf.predict(X_test)
print('Accuracy of SVM classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# #### 3. Decision Trees

# In[120]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 0,max_depth=3, min_samples_leaf=5) 
clf_gini.fit(X_train, y_train) 


# In[121]:


y_pred = clf_gini.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf_gini.score(X_test, y_test)))


# In[122]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", 
            random_state = 0,max_depth=3, min_samples_leaf=5) 
clf_entropy.fit(X_train, y_train)


# In[123]:


y_pred = clf_entropy.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf_entropy.score(X_test, y_test)))


# # Try 2 - Dropping front camera attributes.

# In[124]:


X = train[['battery_power', 'blue', 'clock_speed', 'dual_sim','four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g','touch_screen', 'wifi']]
Y = train["price_range"]


# #### 1. Logistic Regression

# In[125]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[126]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# #### 2. Support Vector Machine

# In[127]:


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='linear') 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
clf.fit(X_train, y_train)


# In[128]:


y_pred = clf.predict(X_test)
print('Accuracy of SVM classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# #### 3. Decision trees

# In[129]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 0,max_depth=3, min_samples_leaf=5) 
clf_gini.fit(X_train, y_train) 


# In[130]:


y_pred = clf_gini.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf_gini.score(X_test, y_test)))


# # Try 3 - Using a feature selector technique

# ### 1. Correlation Statistics

# In[147]:


from sklearn.feature_selection import f_regression


# In[148]:


var_rel = f_regression(X,Y,center = True)


# In[149]:


var_rel


# ### 2. Selection Method

# In[150]:


from sklearn.feature_selection import SelectKBest


# In[153]:


K_Best_Features = SelectKBest(f_regression , k = 5)


# In[154]:


K_Best_Features


# In[155]:


train_new = K_Best_Features.fit_transform(X,Y)


# In[156]:


train_new.shape


# In[158]:


X_new = train_new


# ### Now Lets Use our classifiers again

# #### 1. Logistic regression

# In[159]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.25, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[160]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# #### 2. Support Vector Machine

# In[162]:


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='linear') 
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.25, random_state=0)
clf.fit(X_train, y_train)


# In[163]:


y_pred = clf.predict(X_test)
print('Accuracy of SVM classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# #### 3. Decision tree

# In[164]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size = 0.3, random_state = 0)
clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 0,max_depth=3, min_samples_leaf=5) 
clf_gini.fit(X_train, y_train) 


# In[165]:


y_pred = clf_gini.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf_gini.score(X_test, y_test)))


# # Major Takeaways from above Analysis :
# 
# ### 1. All features are equally important hence no feature selection techniques are used.
# ### 2. Support Vector Machine is the best classifier on this data with an accuracy of above 98 percent.

# ## Classification using SVM without Feature selection and removal.

# In[169]:


predicted_values = clf.predict(test)


# In[170]:


predicted_values


# In[171]:


test["price_range"] = pd.DataFrame(predicted_values)


# In[172]:


test


# ## Saving the final test file with the predicted price range column to a new csv file

# In[173]:


test.to_csv("Mobile Price Classification/Predicted.csv")


# In[ ]:





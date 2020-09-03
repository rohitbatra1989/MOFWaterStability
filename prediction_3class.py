#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import ast
import pickle


# In[8]:

#dataset = 'predict'
dataset = 'validation'
all_fp = pd.read_csv('data/%sdata.csv'%dataset)


# # 3-class Prediction

# NOTE: The following SVM model was trained using Scikit-learn version 0.21.2. Other versions Scikit-learn versions might fail.


model_3class = pickle.load(open("models/svm_ml_3class.pkl", "rb" ) )

# Check if all fingerprint columns used for SVM model exist for new MOFs
for j in range(len(model_3class.Xcols)):
    if model_3class.Xcols[j] in all_fp.columns:
        continue
    else:
        print('Column not found:',model_3class.Xcols[j])
        all_fp[model_3class.Xcols[j]]=0


X = all_fp[model_3class.Xcols].values
prediction = model_3class.predict(X)

prob = model_3class.predict_proba(X)

pred3class = pd.DataFrame(prediction, columns=['3class_prediction'])


# In[10]:


pred3class['prob_class-1'] = prob[:,0]
pred3class['prob_class0'] = prob[:,1]
pred3class['prob_class1'] = prob[:,2]


# In[11]:


pred3class = pd.concat([pred3class, all_fp[['ID','Activated Formula']]], axis=1, sort=False)


# In[12]:

pred3class.to_csv('prediction/3class_%s.csv'%dataset)


# In[ ]:





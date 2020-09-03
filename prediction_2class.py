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


# # 2-class Prediction

# In[21]:


model_2class = pickle.load( open("models/rf_ml_2class.pkl", "rb" ) )

# Check if all fingerprint columns used for SVM model exist for new MOFs
for j in range(len(model_2class.Xcols)):
    if model_2class.Xcols[j] in all_fp.columns:
        continue
    else:
        print('Column not found:',model_2class.Xcols[j],'Adding to fingerprint')
        all_fp[model_2class.Xcols[j]]=0


X = all_fp[model_2class.Xcols].values

prediction = model_2class.predict(X)*-1   # Swapping label, since 2-class model is trained with opposite labels

prob = model_2class.predict_proba(X)
prob[:,[0, 1]] = prob[:,[1, 0]]   # Making 1st column for unstable and 2nd column for stable

pred2class = pd.DataFrame(prediction, columns=['2class_prediction'])
pred2class


# In[22]:


pred2class['prob_class-1'] = prob[:,0]
pred2class['prob_class1'] = prob[:,1]


# In[23]:


pred2class = pd.concat([pred2class, all_fp[['ID','Activated Formula']]], axis=1, sort=False)


# In[24]:


pred2class.to_csv('prediction/2class_%s.csv'%dataset)

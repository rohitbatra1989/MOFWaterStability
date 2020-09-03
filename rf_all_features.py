
# Import libraries
import pandas as pd
import numpy as np
from copy import deepcopy

from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from pycm import ConfusionMatrix

from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import balanced_accuracy_score as bal_score

import matplotlib.pyplot as plt
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier



# Input Params

ntrains = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
seeds = range(10)
savefig = 0

ncvfold = 5

#scoring = 'balanced_accuracy'
scoring = 'f1_macro'
mustXcols = ['L_linkermetalratio', 'L_no','L_noh', 'L_nh2o']


# Load data
ml_onehot_data = pd.read_csv('data/traindata.csv',index_col=None)
ml_onehot_data = ml_onehot_data.sample(frac=1, random_state=1968507).reset_index(drop=True)

# Correct class labels
ml_onehot_data.loc[ml_onehot_data['Stability']==2,'Stability']=-1
ml_onehot_data.loc[ml_onehot_data['Stability']==3,'Stability']=-1
ml_onehot_data.loc[ml_onehot_data['Stability']==0,'Stability']=1
ml_onehot_data.loc[ml_onehot_data['Stability']==1,'Stability']=1

y = ml_onehot_data['Stability']


selXcols = ml_onehot_data.columns[(ml_onehot_data.columns.str.contains('M_')) |
                       (ml_onehot_data.columns.str.contains('L_'))]

assert all(elem in selXcols  for elem in mustXcols)

X = ml_onehot_data[selXcols]


print(X.shape,y.shape)


for ntrain in ntrains:
    for rnseed in seeds:

        print('\n\n#--------------------------------')
        print('Running Code for ntrain: %s, seed: %s'%(ntrain,rnseed))


        # Test train split
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=1-ntrain, random_state=rnseed)


        ###################
        # Random Forest MODEL
        ###################						
        param_grid = {"n_estimators": range(50,150,10), "max_features": range(10,40,5)}
        forest_clf = RandomForestClassifier(class_weight="balanced_subsample", random_state=42,
                                            oob_score=True, n_jobs=-1)

        # Perform Grid search for C and Gamma hyper-parameter        
        clf = GridSearchCV(forest_clf,
                           cv = StratifiedKFold(n_splits=ncvfold,random_state=rnseed, shuffle=True),
                           param_grid = param_grid,
                           scoring=scoring)

        clf.fit(X_train, y_train)

        rf_clf = deepcopy(clf.best_estimator_)

        print("RF params:", len(rf_clf.estimators_))




        ###################
        # Error Evaluations
        ###################

        print("RF Ntrain %s RnSeed %s Train Score %s" %(ntrain, rnseed, rf_clf.score(X_train, y_train)))
        pred = rf_clf.predict(X_train)
        print('Unweighted Accuracy',accuracy(y_pred=pred,y_true=y_train))
        print('Weighted Accuracy',bal_score(y_pred=pred,y_true=y_train))

        cm = ConfusionMatrix(actual_vector=y_train, predict_vector=pred)
        cm.print_matrix()


        print("RF Ntrain %s RnSeed %s Test Score %s" %(ntrain, rnseed, rf_clf.score(X_test, y_test)))
        pred = rf_clf.predict(X_test)
        print('Unweighted Accuracy',accuracy(y_pred=pred,y_true=y_test))
        print('Weighted Accuracy',bal_score(y_pred=pred,y_true=y_test))

        cm = ConfusionMatrix(actual_vector=y_test, predict_vector=pred)
        cm.print_matrix()
        print(cm)

        sys.stdout.flush()

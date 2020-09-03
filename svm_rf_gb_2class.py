
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
savemodel = 0

ncvfold = 5

svm_c_strat = -1.5
svm_c_end = 2.75
svm_c_num = 20

svm_gamma_strat = -3
svm_gamma_end = 1
svm_gamma_num = 20

#scoring = 'balanced_accuracy'
scoring = 'f1_macro'
mustXcols = ['L_linkermetalratio', 'L_no','L_noh', 'L_nh2o']


# Load data
ml_onehot_data = pd.read_csv('data/traindata.csv',index_col=None)
ml_onehot_data = ml_onehot_data.sample(frac=1, random_state=1968507).reset_index(drop=True)


# Correct class labels
# Class 1: Unstable and low-kinetic stability
# Class 0: Stable and high-kinetic stability
# Note: In the manuscript opposite lables have been assigned

ml_onehot_data.loc[ml_onehot_data['Stability']==2,'Stability']=-1
ml_onehot_data.loc[ml_onehot_data['Stability']==3,'Stability']=-1
ml_onehot_data.loc[ml_onehot_data['Stability']==0,'Stability']=1
ml_onehot_data.loc[ml_onehot_data['Stability']==1,'Stability']=1

y = ml_onehot_data['Stability']


# Select RFE based feature list
dimRedu_fnames = ['svmLinear-2class']

for dimRedu_fname in dimRedu_fnames:

	# Select dimReduction X
	selXcols = pd.read_csv('data/selXcols-%s.csv'%dimRedu_fname, index_col=0)
	selXcols = list(selXcols.sel_features)
	print('\n\n#--------------------------------')
	print('\n\n#--------------------------------')
	print('Starting File: ',dimRedu_fname)

	for mustXcol in mustXcols:
		if mustXcol not in selXcols:
			print('Added Xcol: ',mustXcol)
			selXcols.append(mustXcol)

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
			# SVM MODEL
			###################	
			# Seeting grid search for C and gamma hyper-parameter for SVM
			Cs = np.logspace(svm_c_strat, svm_c_end, num=svm_c_num)
			gammas = np.logspace(svm_gamma_strat, svm_gamma_end, num=svm_gamma_num)
			param_grid = {"C": [l for l in Cs],"gamma": [l for l in gammas]}

	        
			# Perform Grid search for C and Gamma hyper-parameter        
			clf = GridSearchCV(svm.SVC(kernel='rbf', probability=True, class_weight='balanced'),
	                           cv = StratifiedKFold(n_splits=ncvfold,random_state=rnseed, shuffle=True),
	                           param_grid = param_grid,
	                           scoring=scoring)
	        
			clf.fit(X_train, y_train)


			# Overall cross-validation score
			print("Ntrain %s RnSeed %s CV Score %s"%(ntrain, rnseed, clf.best_score_))

			# Computing test score
			svm_clf = deepcopy(clf.best_estimator_)

			print("SVM params", svm_clf.gamma, svm_clf.C, svm_clf.class_weight_)

			if savemodel:

				# Saving columns names for use during prediction
				svm_clf.Xcols = selXcols

				# Saving BEST ML Model
				filename = 'svm_ml_2class_ntrain%s_seed%s.pkl'%(ntrain,rnseed)
				pickle.dump(svm_clf, open(filename, 'wb'))			
			


			###################
			# Random Forest MODEL
			###################						
			param_grid = {"n_estimators": range(40,100,5), "max_features": range(3,9)}
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

			if savemodel:

				# Saving columns names for use during prediction
				rf_clf.Xcols = selXcols

				# Saving BEST ML Model
				filename = 'rf_ml_2class_ntrain%s_seed%s.pkl'%(ntrain,rnseed)
				pickle.dump(rf_clf, open(filename, 'wb'))			


			###################
			# Gradient Boost MODEL
			###################	
			param_grid = {"n_estimators": range(2,100,10)}
			gbrt = GradientBoostingClassifier(max_depth=4, learning_rate=.2)

			# Perform Grid search for C and Gamma hyper-parameter        
			clf = GridSearchCV(gbrt,
			                   cv = StratifiedKFold(n_splits=ncvfold,random_state=rnseed, shuffle=True),
			                   param_grid = param_grid,
			                   scoring=scoring)

			clf.fit(X_train, y_train)

			gb_clf = deepcopy(clf.best_estimator_)

			print("GB params", len(gb_clf.estimators_))

			if savemodel:

				# Saving columns names for use during prediction
				gb_clf.Xcols = selXcols

				# Saving BEST ML Model
				filename = 'gb_ml_2class_ntrain%s_seed%s.pkl'%(ntrain,rnseed)
				pickle.dump(gb_clf, open(filename, 'wb'))			



			###################
			# Error Evaluations
			###################						

			#------------------------------------------------
			# Training Errors
			print("SVM Ntrain %s RnSeed %s Train Score %s" %(ntrain, rnseed, svm_clf.score(X_train, y_train)))
			pred = svm_clf.predict(X_train)
			print('Unweighted Accuracy',accuracy(y_pred=pred,y_true=y_train))
			print('Weighted Accuracy',bal_score(y_pred=pred,y_true=y_train))

			cm = ConfusionMatrix(actual_vector=y_train, predict_vector=pred)
			cm.print_matrix()
            

			print("RF Ntrain %s RnSeed %s Train Score %s" %(ntrain, rnseed, rf_clf.score(X_train, y_train)))
			pred = rf_clf.predict(X_train)
			print('Unweighted Accuracy',accuracy(y_pred=pred,y_true=y_train))
			print('Weighted Accuracy',bal_score(y_pred=pred,y_true=y_train))

			cm = ConfusionMatrix(actual_vector=y_train, predict_vector=pred)
			cm.print_matrix()

			print("GB Ntrain %s RnSeed %s Train Score %s" %(ntrain, rnseed, gb_clf.score(X_train, y_train)))
			pred = gb_clf.predict(X_train)
			print('Unweighted Accuracy',accuracy(y_pred=pred,y_true=y_train))
			print('Weighted Accuracy',bal_score(y_pred=pred,y_true=y_train))

			cm = ConfusionMatrix(actual_vector=y_train, predict_vector=pred)
			cm.print_matrix()			
			#------------------------------------------------
			

			#------------------------------------------------
			# Test Errors
			print("SVM Ntrain %s RnSeed %s Test Score %s" %(ntrain, rnseed, svm_clf.score(X_test, y_test)))
			pred = svm_clf.predict(X_test)
			print('Unweighted Accuracy',accuracy(y_pred=pred,y_true=y_test))
			print('Weighted Accuracy',bal_score(y_pred=pred,y_true=y_test))

			cm = ConfusionMatrix(actual_vector=y_test, predict_vector=pred)
			cm.print_matrix()
			print(cm)
			
            
			print("RF Ntrain %s RnSeed %s Test Score %s" %(ntrain, rnseed, rf_clf.score(X_test, y_test)))
			pred = rf_clf.predict(X_test)
			print('Unweighted Accuracy',accuracy(y_pred=pred,y_true=y_test))
			print('Weighted Accuracy',bal_score(y_pred=pred,y_true=y_test))

			cm = ConfusionMatrix(actual_vector=y_test, predict_vector=pred)
			cm.print_matrix()
			print(cm)


			print("GB Ntrain %s RnSeed %s Test Score %s" %(ntrain, rnseed, gb_clf.score(X_test, y_test)))
			pred = gb_clf.predict(X_test)
			print('Unweighted Accuracy',accuracy(y_pred=pred,y_true=y_test))
			print('Weighted Accuracy',bal_score(y_pred=pred,y_true=y_test))

			cm = ConfusionMatrix(actual_vector=y_test, predict_vector=pred)
			cm.print_matrix()
			print(cm)
			#------------------------------------------------

			sys.stdout.flush()  
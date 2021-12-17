import numpy as np
import pandas as pd
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone

import argparse
from sklearn.model_selection import GridSearchCV
import time
import math

np.random.seed(203)

# Loading October dataset 
df = pd.read_csv("/home/Desktop/Mirai_Labeled-October-20-2021.csv", dtype={'ipsrc': str,'Type': str})

## Preprocessing steps on October Data
df=df.drop(['ipsrc','IPsrc_long','IPdst_long','packetNum'],axis=1)

nated_df=df[df['Type'] == 'Nated']

sampled_df=nated_df

Notnated_df=df[df['Type'] == 'Not-Nated']
Notnated_df= Notnated_df.sample(830000)

sampled_df=sampled_df.append(Notnated_df)

y=sampled_df['Type']

sampled_df=sampled_df.drop(['Type'],axis=1)

sampled_df = (sampled_df - np.min(sampled_df, 0)) / (np.max(sampled_df, 0) + 0.0001)

x_train,x_test,y_train,y_test=train_test_split(sampled_df,y,test_size=0.2)

# Loading November Dataset 
df1 = pd.read_csv("/home/Desktop/Mirai_Labeled-November-2021.csv", dtype={'ipsrc': str,'Type': str})

# Preprocessing steps on November Data
df1=df1.drop(['ipsrc','IPsrc_long','IPdst_long','packetNum'],axis=1)

nated_df1=df1[df1['Type'] == 'Nated']
nated_df1= nated_df1.sample(830000)

sampled_df1=nated_df1


Notnated_df1=df1[df1['Type'] == 'Not-Nated']
Notnated_df1= Notnated_df1.sample(830000)

sampled_df1=sampled_df1.append(Notnated_df1)

y1=sampled_df1['Type']

sampled_df1=sampled_df1.drop(['Type'],axis=1)

sampled_df1 = (sampled_df1 - np.min(sampled_df1, 0)) / (np.max(sampled_df1, 0) + 0.0001)


x_train1,x_test1,y_train1,y_test1=train_test_split(sampled_df1,y1,test_size=0.2)

# perform a grid search on the 'C' parameter of Logistic Regression
print("SEARCHING LOGISTIC REGRESSION")
params = {"C": [1.0, 1000.0, 6000.0, 10000.0]}
start = time.time()
gs = GridSearchCV(linear_model.LogisticRegression(), params, n_jobs = -1, verbose = 1)
gs.fit(x_train, y_train)
# print diagnostic information to the user and grab the best model
print("done in %0.3fs" % (time.time() - start))
print("best score: %0.3f" % (gs.best_score_))
print("LOGISTIC REGRESSION PARAMETERS")
bestParams = gs.best_estimator_.get_params()
# loop over the parameters and print each of them out so they can be manually set
for p in sorted(params.keys()):
 print("\t %s: %f" % (p, bestParams[p]))

# Logistic Regression CLassifier 
time_start = time.time()
logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)                                
logistic.C = 1

logistic.fit(x_train, y_train)
print('Logistic regression done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
## Testing on November Data
y_pred = logistic.predict(x_test1)

print('Flow Classification Time or Predict Time: {} seconds'.format(time.time()-time_start))
print("Logistic regression:\n%s\n" % (
       metrics.classification_report(y_test1, y_pred)))

#perform a grid search on the learning rate, number of
#iterations, and number of components on the RBM and
# C for Logistic Regression
print ("SEARCHING RBM + LOGISTIC REGRESSION")
params = {
 "rbm__learning_rate": [0.06, 0.01, 0.001],
 "rbm__n_components": [50, 100, 200],
 "logistic__C": [1.0, 6000.0, 10000.0]}
# perform a grid search over the parameter
start = time.time()
gs = GridSearchCV(rbm_features_classifier, params, n_jobs = -1, verbose = 1)
gs.fit(x_train, y_train)
# print diagnostic information to the user and grab the best model
print ("\ndone in %0.3fs" % (time.time() - start))
print ("best score: %0.3f" % (gs.best_score_))
print ("RBM + LOGISTIC REGRESSION PARAMETERS")
bestParams = gs.best_estimator_.get_params()
# loop over the parameters and print each of them out
# so they can be manually set
for p in sorted(params.keys()):
 print ("\t %s: %f" % (p, bestParams[p]))

## Logistic Regression with RBM
time_start = time.time()
logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1) 
                                           
rbm = BernoulliRBM(random_state=0, verbose=True)

rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('logistic', logistic)])

rbm.learning_rate = 0.001
rbm.n_components = 200
logistic.C = 10000

rbm_features_classifier.fit(x_train, y_train)
print('Logistic regression using RBM features done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
## Testing on November data
y_pred = rbm_features_classifier.predict(x_test1)

print('Flow Classification Time or Predict Time: {} seconds'.format(time.time()-time_start))
print("Logistic regression with RBM:\n%s\n" % (
       metrics.classification_report(y_test1, y_pred)))

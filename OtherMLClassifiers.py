import numpy as np
import pandas as pd
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import argparse
import time
import math
import lightgbm as lgb

np.random.seed(203)

#Loading October Dataset
df = pd.read_csv("/home/Desktop/Mirai_Labeled-October-20-2021.csv", dtype={'ipsrc': str,'Type': str})

df=df.drop(['packetNum','ipsrc','IPsrc_long','IPdst_long'],axis=1)

##Dropping least important features for the RF model 
##df=df.drop(['packetNum','ipsrc','IPsrc_long','IPdst_long', 'tcp_flag',
##             'tcpdatalen','prtcl','timestamp','tcp_ack_seq','tcp_urp','TCP_OPT_SACK','tcp_reserve'],axis=1)

nated_df=df[df['Type'] == 'Nated']

sampled_df=nated_df

Notnated_df=df[df['Type'] == 'Not-Nated']
Notnated_df= Notnated_df.sample(830000)

sampled_df=sampled_df.append(Notnated_df)

y=sampled_df['Type']

sampled_df=sampled_df.drop(['Type'],axis=1)

sampled_df = (sampled_df - np.min(sampled_df, 0)) / (np.max(sampled_df, 0) + 0.0001)

x_train,x_test,y_train,y_test=train_test_split(sampled_df,y,test_size=0.2)

#Loading November Dataset
df1 = pd.read_csv("/home/Desktop/Mirai_Labeled-November-2021.csv", dtype={'ipsrc': str,'Type': str})

df1=df1.drop(['ipsrc','IPsrc_long','IPdst_long','packetNum'],axis=1)

##Dropping least important features for the RF model 
##df1=df1.drop(['packetNum','ipsrc','IPsrc_long','IPdst_long', 'tcp_flag',
##             'tcpdatalen','prtcl','timestamp','tcp_ack_seq','tcp_urp','TCP_OPT_SACK','tcp_reserve'],axis=1)

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


#LGBM classifier
time_start = time.time()

clf = lgb.LGBMClassifier()
clf.fit(x_train, y_train)

print('LightGBM classifier done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
#Testing on November Data
y_pred=clf.predict(x_test1)

print('Flow Classification Time or Predict Time: {} seconds'.format(time.time()-time_start))
print("LightGBM classifier:\n%s\n" % (
    metrics.classification_report(y_test1, y_pred)))

#Linear SVM Classifier
time_start = time.time()

clf = LinearSVC()
clf.fit(x_train, y_train)

print('LinearSVC classifier done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
#Testing on November Data
y_pred=clf.predict(x_test1)

print('Flow Classification Time or Predict Time: {} seconds'.format(time.time()-time_start))
print("LinearSVC classifier:\n%s\n" % (
    metrics.classification_report(y_test1, y_pred)))

#Random Forest Classifier
time_start = time.time()

clf = RandomForestClassifier()
clf.fit(x_train, y_train)

print('Random Forest classifier done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
#Testing on November Data
y_pred=clf.predict(x_test1)

print('Flow Classification Time or Predict Time: {} seconds'.format(time.time()-time_start))
print("Random Forest classifier:\n%s\n" % (
    metrics.classification_report(y_test1, y_pred)))


#Feature Importance
feature_scores = pd.Series(clf.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print(feature_scores)

#GaussianNB Classifier
time_start = time.time()

clf = GaussianNB()
clf.fit(x_train, y_train)

print('GaussianNB classifier done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
#Testing on November Data
y_pred=clf.predict(x_test1)

print('Flow Classification Time or Predict Time: {} seconds'.format(time.time()-time_start))
print("GaussianNB classifier:\n%s\n" % (
    metrics.classification_report(y_test1, y_pred)))

#MLP Classifier
time_start = time.time()

clf = MLPClassifier(activation='tanh',solver='sgd')
clf.fit(x_train, y_train)

print('MLP classifier done! Time elapsed: {} seconds'.format(time.time()-time_start))

time_start = time.time()
#Testing on November Data
y_pred=clf.predict(x_test1)

print('Flow Classification Time or Predict Time: {} seconds'.format(time.time()-time_start))
print("MLP classifier:\n%s\n" % (
    metrics.classification_report(y_test1, y_pred)))

from pytorch_tabnet.tab_model import TabNetClassifier

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
np.random.seed(0)


import os
import wget
from pathlib import Path

dataset_name = 'NATed Mirai'

# Loading October Dataset
Train = pd.read_csv("/home/Desktop/Mirai_Labeled-October-20-2021.csv", dtype={'ipsrc': str,'Type': str})

## Preprocessing steps on October Data
Train=Train.drop(['ipsrc','IPsrc_long','IPdst_long','packetNum'],axis=1)

nated_df=Train[Train['Type'] == 'Nated']
nated_df= nated_df.sample(830000)

sampled_df=nated_df

Notnated_df=Train[Train['Type'] == 'Not-Nated']
Notnated_df= Notnated_df.sample(830000)

sampled_df=sampled_df.append(Notnated_df)

train=sampled_df.reset_index()

# Loading November Data
df = pd.read_csv("/home/Desktop/Mirai_Labeled-November-2021.csv", dtype={'ipsrc': str,'Type': str})

#Preprocessing steps on November Data
df=df.drop(['ipsrc','IPsrc_long','IPdst_long','packetNum'],axis=1)

nated_df=df[df['Type'] == 'Nated']
nated_df= nated_df.sample(830000)

sampled_df=nated_df

Notnated_df=df[df['Type'] == 'Not-Nated']
Notnated_df= Notnated_df.sample(830000)

sampled_df=sampled_df.append(Notnated_df)

train_november=sampled_df.reset_index()

train_november["Set"]= 'test'

target = 'Type'

# Dividing October Dataset into training, validation, and testing sets 
##if "Set" not in train.columns:
##    train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

# Dividing October Data into training and validation sets
if "Set" not in train.columns:
 train["Set"] = np.random.choice(["train", "valid"], p =[.8, .2], size=(train.shape[0],))

#Appending November dataset as a testing set
train=train.append(train_november).reset_index()

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index
test_indices = train[train.Set=="test"].index

nunique = train.nunique()
types = train.dtypes

categorical_columns = []
categorical_dims =  {}
for col in train.columns:
    if types[col] == 'object' or nunique[col] < 200:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)


unused_feat = ['Set']

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

tabnet_params = {"cat_idxs":cat_idxs,
                 "cat_dims":cat_dims,
                 "cat_emb_dim":1,
                 "optimizer_fn":torch.optim.Adam,
                 "optimizer_params":dict(lr=2e-2),
                 "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                 "gamma":0.9},
                 "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                 "mask_type":'sparsemax'
                }

clf = TabNetClassifier(**tabnet_params
                      )
X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]

max_epochs = 100 if not os.getenv("CI", False) else 2

# This illustrates the warm_start=False behaviour
save_history = []
for _ in range(2):
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['accuracy'],
        max_epochs=max_epochs , patience=20,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False
    )
    save_history.append(clf.history["valid_accuracy"])

y_pred=clf.predict(X_test)
test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
print(f"FINAL TEST SCORE FOR {dataset_name} : {test_acc}")

# save tabnet model
saving_path_name = "/home//Desktop/tabnet_model_accuracy"
saved_filepath = clf.save_model(saving_path_name)

# define new model with basic parameters and load state dict weights
loaded_clf = TabNetClassifier()
loaded_clf.load_model(saved_filepath)

loaded_y_pred=loaded_clf.predict(X_test)
loaded_test_acc = accuracy_score(y_pred=loaded_y_pred, y_true=y_test)
print(f"FINAL TEST SCORE FOR {dataset_name} : {loaded_test_acc}")

#Feature Importance
print(clf.feature_importances_)

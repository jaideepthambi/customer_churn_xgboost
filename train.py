# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 17:53:58 2021

@author: JaideepThambi
"""

from sklearn.model_selection import train_test_split
import featuretools as ft
import pandas as pd
import xgboost as xgb


def train_xgb(X_train, X_test, y_train, y_test):

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    evals = [(dtrain, 'train'), (dvalid, 'valid')]

    params = {
        'min_child_weight': 1, 'eta': 0.166,
        'colsample_bytree': 0.4, 'max_depth': 9,
        'subsample': 1.0, 'lambda': 57.93,
        'booster': 'gbtree', 'gamma': 0.5,
        'verbosity': 1, 'eval_metric': 'auc',
        'objective': 'binary:logistic',
    }

    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=227,
                      evals=evals, early_stopping_rounds=60, maximize=False,
                      verbose_eval=10)

    print('Modeling AUC  %.5f' % model.best_score)
    return model

import numpy as np
import pandas as pd
import os
cwd = os.getcwd()
pd.options.display.max_columns = 25
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
import xgboost as XGB
import itertools
from sklearn.metrics import accuracy_score
from dython import nominal
from sklearn.model_selection import GridSearchCV


def tune_param(x_train, y_train):
    print("Starts to Tune Parameters !")
    models = []
# Get the best parameter of our Models
    random_state = 42
#Logistic Regession
    par_lg = [{"penalty": ["l2"], "C": [0.1, 1, 10], "solver": ["lbfgs"]},
              {"penalty": ["l1", "l2"], "C": [0.1, 1, 10], "solver": ["liblinear"]}]
    lg = LogisticRegression(class_weight = "balanced")
    grid = GridSearchCV(lg, par_lg, cv = 5, return_train_score = True, scoring = "roc_auc")
    grid = grid.fit(x_train, y_train)
    print("Best Param (LogReg): ", grid.best_params_)
    print("Best Score (LogReg): ", grid.best_score_)
    lg = grid.best_estimator_
    models.append(lg)

#Support Vector Machine
    par_svm = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["auto", "scale"]}
    svm = SVC(class_weight = "balanced")
    grid = GridSearchCV(svm, par_svm, cv = 5, return_train_score = True, scoring = "roc_auc")
    grid = grid.fit(x_train, y_train)
    print("Best Param (SVM): ", grid.best_params_)
    print("Best Score (SVM): ", grid.best_score_)
    svm = grid.best_estimator_
    models.append(svm)

#Random Forest
    par_rf = {"n_estimators": [5, 10, 15, 20, 50, 100], "max_depth": [2, 5, 8, 10]}
    rf = RandomForestClassifier(class_weight = "balanced")
    grid = GridSearchCV(rf, par_rf, cv = 5, return_train_score = True, scoring = "roc_auc")
    grid = grid.fit(x_train, y_train)
    print("Best Param (RF): ", grid.best_params_)
    print("Best Score (RF): ", grid.best_score_)
    rf = grid.best_estimator_
    models.append(rf)
#XGBoost
    par_xgb = {"n_estimators": [5, 10, 15, 20, 50, 100], "max_depth": [2, 5, 8, 10, 15]}
    xgb = XGB.XGBClassifier(class_weight = "balanced")
    grid = GridSearchCV(xgb, par_xgb, cv = 5, return_train_score = True, scoring = "roc_auc")
    grid = grid.fit(x_train, y_train)
    print("Best Param (xgb): ", grid.best_params_)
    print("Best Score (xgb): ", grid.best_score_)
    xgb = grid.best_estimator_
    models.append(xgb)
#Extratree
    par_extra = {"n_estimators": [5, 10, 15, 20, 50 , 100], "max_depth": [2, 5, 8, 10, 15]}
    extra = ExtraTreesClassifier(class_weight = "balanced")
    grid = GridSearchCV(extra, par_extra, cv = 5, return_train_score = True, scoring = "roc_auc")
    grid = grid.fit(x_train, y_train)
    print("Best Param (extra): ", grid.best_params_)
    print("Best Score (extra): ", grid.best_score_)
    extra = grid.best_estimator_
    models.append(extra)

    return models

    


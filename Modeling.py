import numpy as np
import pandas as pd
import os
cwd = os.getcwd()
pd.options.display.max_columns = 25
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit,learning_curve, GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score
from dython import nominal
from ClassWeights import tune_param

sns.set(font_scale = 1)

#split label
x = pd.read_csv(os.path.join(cwd, "data/ReadyForTrain.csv"))
y_train = x["status_id_binary"]

cate_col = ['payment_plan', 'program_name', 'application_type_name', 
            'referrer', 'gender', 'home_country', 'work_country', 
            'practice_type', 'professional_assoc', 'home_state', 
            'work_state', 'Orientation']

nume_col = ["response", "hours_online", 'preprobation', 'currentafterpreprbation', "Unit 1", "Unit 2", "Unit 3",
            "Unit 4"]

# Numerical Heatmap
#nominal.associations(x[nume_col])
# Total Heatmap
nominal.associations(x, nominal_columns = cate_col)

# Drop the unnecessary column
drop_column = ["Unit 1", "Unit 2", "Unit 3", "Unit 4", "status_id_binary", "user_id", "application_id", "gender", "application_type_name", "home_country", "home_state", "professional_assoc"]
tmp = x.drop(columns =drop_column)
# One Hot-Encoding
cate_col_new = [x for x in cate_col if x not in drop_column]
x_train = pd.get_dummies(data = tmp, columns = cate_col_new)
columns = list(x.columns)

# Data Split
train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size = 0.25, random_state = 42, stratify = y_train)

#Tune parameters
models = tune_param(train_x, train_y)
lg = models[0]
svm = models[1]
rf = models[2]
xgb = models[3]
extra = models[4]

# Cross Validation of All Models
random_state = 42
kfold = StratifiedShuffleSplit(n_splits=5, random_state = random_state)

cv_results = []
for classifier in models :
    cv_results.append(cross_val_score(classifier, train_x, y = train_y, scoring = "f1", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["LogisticRegression", "SVM",
                                                                                        "RandomForest", "XGBoost", 
                                                                                        "ExtraTrees"]})

fig, ax = plt.subplots(figsize = (13, 8))
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std}, ax=ax)
g.set_xlabel("Mean Score", fontsize = 30)
g = g.set_title("Cross validation scores", fontsize = 30)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.xlim(0.5, 1.1)
plt.show()

# Learning Curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
     
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

accuracy_all = []

print("========== Logistic Regression ==========")
title = "Learning Curves (Logistic Regression)"
lg.fit(x_train, y_train)
plot_learning_curve(lg, title, x_train, y_train, ylim=(0.1, 1.01), cv=kfold, n_jobs=4)
plt.show()

#Confusion Matrix
lg.fit(train_x, train_y)
pred_lg = lg.predict(test_x)
accuracy_all.append(accuracy_score(test_y, pred_lg))

cf_lg = confusion_matrix(test_y, pred_lg)
print("Confusion Matrix (LogisticRegression): \n", cf_lg)
print(classification_report(test_y, pred_lg))
fpr_lg, tpr_lg, _ = roc_curve(test_y, pred_lg)


print("========== SVM ==========")
title = "Learning Curves (SVM)"
svm.fit(x_train, y_train)
plot_learning_curve(svm, title, x_train, y_train, ylim=(0.1, 1.01), cv=kfold, n_jobs=4)
plt.show()

#Confusion Matrix
svm.fit(train_x, train_y)
pred_svm = svm.predict(test_x)
accuracy_all.append(accuracy_score(test_y, pred_svm))

cf_svm = confusion_matrix(test_y, pred_svm)
print("Confusion Matrix (SVM): \n", cf_svm)
print(classification_report(test_y, pred_svm))
fpr_svm, tpr_svm, _ = roc_curve(test_y, pred_svm)


print("========== Random Forest ==========")
title = "Learning Curves (RF)"
rf.fit(x_train, y_train)
plot_learning_curve(rf, title, x_train, y_train, ylim=(0.1, 1.01), cv=kfold, n_jobs=4)
plt.show()

#Confusion Matrix
rf.fit(train_x, train_y)
pred_rf = rf.predict(test_x)
accuracy_all.append(accuracy_score(test_y, pred_rf))

cf_rf = confusion_matrix(test_y, pred_rf)
print("Confusion Matrix (RandomForest): \n", cf_rf)
print(classification_report(test_y, pred_rf))
fpr_rf, tpr_rf, _ = roc_curve(test_y, pred_rf) 


print("========== XGBoost ==========")
title = "Learning Curves (XGBoost)"
xgb.fit(x_train, y_train)
plot_learning_curve(xgb, title, x_train, y_train, ylim=(0.1, 1.01), cv=kfold, n_jobs=4)
plt.show()

#Confusion Matrix
xgb.fit(train_x, train_y)
pred_xgb = xgb.predict(test_x)
accuracy_all.append(accuracy_score(test_y, pred_xgb))

cf_xgb = confusion_matrix(test_y, pred_xgb)
print("Confusion Matrix (XGBoost): \n", cf_xgb)
print(classification_report(test_y, pred_xgb))
fpr_xgb, tpr_xgb, _ = roc_curve(test_y, pred_xgb)


print("========== ExtraTrees ==========")
title = "Learning Curves (ExtraTrees)"
extra.fit(x_train, y_train)
plot_learning_curve(extra, title, x_train, y_train, ylim=(0.1, 1.01), cv=kfold, n_jobs=4)
plt.show()

#Confusion Matrix
extra.fit(train_x, train_y)
pred_extra = extra.predict(test_x)
accuracy_all.append(accuracy_score(test_y, pred_extra))

cf_extra = confusion_matrix(test_y, pred_extra)
print("Confusion Matrix (ExtraTrees): \n", cf_extra)
print(classification_report(test_y, pred_extra))
fpr_extra, tpr_extra, _ = roc_curve(test_y, pred_extra)


# ROC curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_lg, tpr_lg, label='LR')
plt.plot(fpr_svm, tpr_svm, label='SVM')
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')
plt.plot(fpr_extra, tpr_extra, label = 'ExtraTrees')
plt.xlabel('False positive rate', fontsize = 20)
plt.ylabel('True positive rate', fontsize = 20)
plt.title('ROC curve', fontsize = 20)
plt.legend(loc='best', fontsize = 20)
plt.show()

# Test Prediction Accuracy
fig, ax = plt.subplots(figsize = (13, 8))
acc_res = pd.DataFrame({"AccuracyAll":accuracy_all,
        "Algorithm":["LogisticRegression", "SVM", "RandomForest",
                     "XGBoost", "ExtraTrees"]})

g = sns.barplot("AccuracyAll","Algorithm",data = acc_res, palette="Set3",orient = "h", ax = ax)
g.set_xlabel("Accuracy", fontsize = 30)
g = g.set_title("Prediction Score", fontsize = 30)
plt.xlim(0.5, 1.1)
plt.yticks(fontsize = 30)
plt.xticks(fontsize = 30)
plt.show()

# Feature Importance Table
nrows = 3
ncols = 1
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,9), squeeze = False)

names_classifiers = [("RandomForest", rf), ("XGBoost", xgb), ("ExtraTrees", extra)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:5]
        g = sns.barplot(y = train_x.columns[indices][:5], x = classifier.feature_importances_[indices][:5] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=20)
        if ncols == 1:
            g.set_ylabel("Features",fontsize=20)
        g.tick_params(labelsize=20)
        g.set_title(name + " feature importance", fontsize = 20)
        nclassifier += 1

plt.show()


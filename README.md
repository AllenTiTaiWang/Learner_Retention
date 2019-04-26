# Learner Retention

In order to better understand the learner retention in IHeLp program, 
a predictive analysis were conducted on IHeLp data.

it contains:

1. Preprocessing
2. Exploratory data analysis
3. feature engineering
4. Modeling
5. Evaluation

## Overview of Data

272 students with 20 features.

| Individual Characteristics | Behavioral Features |
| --- | --- |
| Payment plan             | Number of responses |
| Program name             | Houes online        |
| Application type         | Pre-probation       |
| Home (work) state        | Probation           |
| Home (work) country      | Current after pre-probation |
| Gender                   | Orientation         |
| Practice type            | Final exam          |
| Professional association | Amounts of hour on units |
| Referrer                 | Status (label)      |

## Getting Started

Use the script in sql file to get two original data tables from the database of AW center

### Prerequisites

```
python>=3.72
numpy>=1.15
scikit-learn>=0.20
```

### Installing

Clone this repository

```
git clone https://allentitaiwang@bitbucket.org/azcim/analytics.git
```

### Going to the right Directory

First of all, go to EnrollDec file.

```
cd analytics/learner_retention/EnrollDec
```

## Pipeline

### Preprocessing

Before modeling, we have to transfer what we have into feature matrix and label serie.

Firstly, go to Preprocessing file.

```
cd Preprocessing
```

and then run the following scripts.

```
python3 FiltWID.py
python3 CombineUSTUTR.py
python3 Pivot_table.py
python3 BehavioralAdjust.py
```

These codes will combine the tables into the format we need for modeling.
And all of the output from each script can be found in data file.

### Exploratory Data Analysis

In this step, you can see multiple graphs of each feature in the data.

```
python3 DataExplore.py
```

For example,

#### Payment Plan

The bar plot of students in different payment plan.

![alt text](https://github.com/AllenTiTaiWang/Learner_Retention/blob/master/pics/Payment.png)

#### learner Performance

The scatter plot of students with different status label and the spending hours on units.

![alt text](https://github.com/AllenTiTaiWang/Learner_Retention/blob/master/pics/units.png)

### Feature Engineering

Before Modeling, we can adjust our features by observing Heatmap.

```
python3 Modeling
```

![alt text](https://github.com/AllenTiTaiWang/Learner_Retention/blob/master/pics/Heatmap.png)

### Modeling

In this step, learning curves and cross validation score can tell us how to adjust our models.

For instance,

Logistic Regression model

![alt text](https://github.com/AllenTiTaiWang/Learner_Retention/blob/master/pics/LogReg.png)

Random Forest model

![alt text](https://github.com/AllenTiTaiWang/Learner_Retention/blob/master/pics/RF.png)

Cross Validation Score of models

![alt text](https://github.com/AllenTiTaiWang/Learner_Retention/blob/master/pics/CV_score.png)


### Evaluation

To further look into F1-score, precision, and recall.

ROC curve

![alt text](https://github.com/AllenTiTaiWang/Learner_Retention/blob/master/pics/ROC.png)


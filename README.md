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

![alt text](https://bitbucket.org/azcim/analytics/src/master/learner_rentention/EnrollDec/pics/Payment.png)

#### learner Performance

The scatter plot of students with different status label and the spending hours on units.

![alt text](https://bitbucket.org/azcim/analytics/src/master/learner_rentention/EnrollDec/pics/units.png)

### Feature Engineering

Add additional notes about how to deploy this on a live system

### Modeling

### Evaluation



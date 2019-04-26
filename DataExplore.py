import numpy as np
import pandas as pd
import os
cwd = os.getcwd()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "ticks", color_codes = True)
from math import log

#Data input
train = pd.read_csv(os.path.join(cwd, "data/PreprocessingData/UTotal.csv"))

#Set display
pd.options.display.max_columns = 25
sns.set(font_scale = 2)

#Data Exploration
train.info()
print("---------")
print(train.describe())

#Check Null percentage
print(train.isnull().mean().round(4) * 100)
#sns.catplot(x = "status_id", kind = "count", data = train)
#plt.show()




#Total 263
#status_id = label (multiple) 263 - ID all in W
print("status_id ----------")
print(train.groupby("status_id")["application_id"].nunique())
train["status_id_binary"] = train["status_id"].map({"I": 0, "D": 0, "W": 0, "G": 1})

f,ax=plt.subplots(1,2,figsize=(18,8))
train['status_id_binary'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Graduated', fontsize = 30)
ax[0].set_ylabel('')
sns.countplot('status_id_binary',data=train,ax=ax[1])
ax[1].set_title('Graduated', fontsize = 30)
plt.show()

#num_payments are all the smae 255
print(train.groupby("payment_plan")["application_id"].nunique())

train["payment_plan"] = train["payment_plan"].fillna("Other")
#train["payment_plan"] = train["payment_plan"].map({"1 Payment": 3, "2 Payments": 2, "6 monthly payments": 1,
#                                                    "6 Monthly Payments": 1, "Full Scholarship": 4, "Other": 0})
train["payment_plan"] = train["payment_plan"].replace({"6 Monthly Payments": "6 monthly payments"})
fig, ax = plt.subplots(figsize = (13, 8))
ax.set_title('Payment Plan')
sns.catplot(x = "status_id_binary", kind = "count", hue = "payment_plan", data = train, ax = ax)
plt.show()

#num_posts 263
#num_login_days 263
#program_name 263
print("program_name ----------")
print(train.groupby("program_name")["application_id"].nunique())
train["program_name"] = train["program_name"].map({"IHeLp 2014 Summer": "Summer", "IHeLp 2014 Winter": "Spring",
                                                    "IHeLp 2015 Summer": "Summer", "IHeLp 2016 Spring": "Spring", 
                                                    "IHeLp 2016 Summer": "Summer", "IHeLp 2017 Spring": "Spring", 
                                                    "IHeLp 2017 Summer": "Summer", "IHeLp 2018 Spring": "Spring", 
                                                    "IHeLp 2018 Summer": "Summer", "IHeLp 2019 Summer": "Summer", 
                                                     })
print(train.groupby("program_name")["application_id"].nunique())
sns.catplot(x = "status_id_binary", kind = "count", hue = "program_name", data = train)
plt.show()

#application_type_name (3 classes inside) 263
train["application_type_name"] = train["application_type_name"].replace({"Regular Discounted": "Discounted"})
sns.catplot(x = "status_id_binary", kind = "count", hue = "application_type_name", data = train)
plt.show()
#professional_assoc (multiple) 263
print("professional_assoc ----------")
print(train.groupby("professional_assoc")["application_id"].nunique())
train["professional_assoc"] = train["professional_assoc"].map(lambda x: "RegisteredNurse" if x == "Registered Nurse" else "Others")
sns.catplot(x = "status_id_binary", kind = "count", hue = "professional_assoc", data = train)
plt.show()

#referer 259
print("referrer ----------")
print(train.groupby("referrer")["application_id"].nunique())
train["referrer"] = train["referrer"].fillna("Other")
train["referrer"] = train["referrer"].replace({"AF member": "Associate", "Brochure/Mailing": "Ad", "CIM faculty": "Associate", 
                                                "DrWeil": "DrWeil", "DrWeil_event": "DrWeil", "Healthy_Aging": "Other", 
                                                "Prevention": "Other", "Web": "Ad"})
print(train.groupby("referrer")["application_id"].nunique())
sns.catplot(x = "status_id_binary", kind = "count", hue = "referrer", data = train)
plt.show()

#gender 259
print("gender ----------")
train["gender"] = train["gender"].fillna("F")
print(train.groupby("gender")["application_id"].nunique())
#practice_type
print("practice_type ---------------")
print(train.groupby("practice_type")["application_id"].nunique())
train["practice_type"] = train["practice_type"].fillna("Other")
train["practice_type"] = train["practice_type"].map({"Acupuncturist": 1, "Cardiology": 1, "Emergency Medicine": 1,
                                                    "Fam_Prac": 1, "Family Medicine": 1, "Int_med": 1, "OB/Gyn": 1,
                                                    "Other": 0, "Pain Management": 1, "Pediatrics": 1, "Public Health &amp; Preventive Medicine": 1, 
                                                    "Rheumatology": 1, "Specialty Not Listed": 0, "Sub": 1, "Surgery": 1})
print(train.groupby("practice_type")["application_id"].nunique())
train["practice_type"] = train["practice_type"].map(lambda x: "Listed" if x == 1 else "NotListed")
sns.catplot(x = "status_id_binary", kind = "count", hue = "practice_type", data = train)
fig, ax = plt.subplots(figsize = (13, 8))
sns.catplot(x = 'referrer', y = 'status_id_binary', hue = 'practice_type', col = 'professional_assoc', kind = 'bar', data = train)
plt.show()

#home country 215
print("home_country ----------")
train["home_country"] = train["home_country"].fillna("Unknown")
print(train.groupby("home_country")["application_id"].nunique())
train["home_country"] = train["home_country"].replace({"Australia": 0, "Brazil": 0, "Canada": 0, "United Arab Emirates": 0,
                                                    "Greentownship": 0, "Japan": 0, "Pima": 0, "Scotland": 0,  
                                                    "US":1, "USA": 1, "United Sta": 1, "United States": 1, 
                                                    "United States ": 1, "United States of America (the)": 1, 
                                                    "usa": 1})
print(train.groupby("home_country")["application_id"].nunique())
train["home_country"] = train["home_country"].map(lambda x: "US" if x == 1 else ("Unknown" if x == "Unknown"else "Intl"))

#work country 206
print("work_country ----------")
train["work_country"] = train["work_country"].fillna("Unknown")
print(train.groupby("work_country")["application_id"].nunique())
train["work_country"] = train["work_country"].replace({"Australia": 0, "Brazil": 0, "Canada": 0, "United Arab Emirates": 0,
                                                    "United Kingdom": 0, "Japan": 0, "Pima": 0, "Costa Rica": 0,  
                                                    "US":1, "USA": 1, "United Sta": 1, "United States": 1, 
                                                    "United States ": 1, "United States of America (the)": 1, 
                                                    "usa": 1})
print(train.groupby("work_country")["application_id"].nunique())
train["work_country"] = train["work_country"].map(lambda x: "US" if x == 1 else ("Unknown" if x == "Unknown" else "Intl"))

#home sate
train["home_state"] = train["home_state"].fillna("Unknown")
print(train.groupby("home_state")["application_id"].nunique())
train["home_state"] = train["home_state"].map(lambda x: "AZ&CA" if x == ("AZ" or "CA") else ("Unknown" if x == "Unknown" else "OthStates"))
print(train.groupby("home_state")["application_id"].nunique())

#work state
train["work_state"] = train["work_state"].fillna("Unknown")
print(train.groupby("work_state")["application_id"].nunique())
train["work_state"] = train["work_state"].map(lambda x: "AZ&CA" if x == ("AZ" or "CA") else ("Unknown" if x == "Unknown" else "OthStates"))
print(train.groupby("work_state")["application_id"].nunique())
fig, ax = plt.subplots(1, 2, figsize = (13, 8))
sns.catplot(x = "home_state", y = "status_id_binary", kind = "bar", hue = "work_state", data = train, ax=ax[0])
sns.catplot(x = "home_country", y = "status_id_binary", kind = "bar", hue = "work_country", data = train, ax=ax[1])
plt.show()

#unit1-unit6 263 SUM
unit_list = ["Unit 1", "Unit 2", "Unit 3", "Unit 4"]
train["units"] = train[unit_list].sum(axis = 1)
train["Unit 1"] = train["Unit 1"].fillna(0)
train["Unit 2"] = train["Unit 2"].fillna(0)
train["Unit 3"] = train["Unit 3"].fillna(0)
train["Unit 4"] = train["Unit 4"].fillna(0)
countz = (train[unit_list] != 0).sum(axis = 1)
print(countz)
train["units"] = train.apply(lambda x: 0 if x["units"] == 0 else round(x["units"] / countz, 2), axis = 1)
#train["num_posts"] = round(train["num_posts"] / countz, 2)
train["response"] = train.apply(lambda x: 0 if x["units"] == 0 else round(x["response"] / countz, 2), axis = 1)
#train["num_login_days"] = round(train["num_login_days"] / countz, 2)
train["hours_online"] = train.apply(lambda x: 0 if x["units"] == 0 else round(np.log(x["hours_online"] / countz), 2), axis = 1)


sns.catplot(x = "status_id_binary", y = "hours_online", data = train, hue = "preprobation")
plt.show()
sns.catplot(x = "status_id_binary", y = "response", data = train, hue = "preprobation")
plt.show()
sns.catplot(x = "status_id_binary", y = "units", data = train, hue = "preprobation")
plt.show()



#Select Final Features
train_out = train[["user_id", "application_id", "payment_plan", "response", "hours_online", "program_name", 
                    "application_type_name", "referrer", "gender", "home_country", "work_country", "practice_type"
                    , "professional_assoc", "home_state", "work_state", "preprobation", "currentafterpreprbation", "Orientation", "Unit 1", "Unit 2", 
                    "Unit 3", "Unit 4", "units", "status_id_binary"]]
#print(train_out)
#Mapping done
train_out.to_csv("data/ReadyForTrain.csv", sep = ",", index = False)

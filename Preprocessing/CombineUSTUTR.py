import pandas as pd
import os
cwd = os.getcwd()
import glob


UST = pd.read_csv(os.path.join(cwd, "../data/PreprocessingData/UST_cleaned.csv"))
print("UST Length: ",UST.shape[0])
UTR = pd.read_csv(os.path.join(cwd, "../data/PreprocessingData/UTR_cleaned.csv"))
print("UTR Length: ", UTR.shape[0])

# make ditionary
dic_s = set()
dic = {}
for i in range(UTR.shape[0]):
    user = UTR.iloc[i]['user_id']
    curriculum = UTR.iloc[i]['curriculum_id']
    dic_s.add((user, curriculum))
    dic[(user, curriculum)] = UTR.iloc[i]['sum']
#print(dic_s)

check = pd.Series([])
for i in range(UST.shape[0]):
    user = UST.iloc[i]['user_id']
    curriculum = UST.iloc[i]['curriculum_id']
    item = (user, curriculum)
    if item in dic_s:
        check = check.append(pd.Series([dic[(user, curriculum)]]))
    else:
        check = check.append(pd.Series([0]))

UST = UST.assign(check = check.values)
print("UST Length: ", UST.shape[0])
UST.to_csv("../data/PreprocessingData/Ucom.csv", sep = ",", index = False)

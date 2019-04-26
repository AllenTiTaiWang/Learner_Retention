import pandas as pd
import os
cwd = os.getcwd()
import glob


UST = pd.read_csv(os.path.join(cwd, "../data/RawData/UserShouldTake.csv"))
print("UST Length: ",UST.shape[0])
UTR = pd.read_csv(os.path.join(cwd, "../data/RawData/UserTakeReq.csv"))
print("UTR Length: ", UTR.shape[0])
URW = pd.read_csv(os.path.join(cwd, "../../IHeLp/withdrawn_apps.csv"))

URWL = []
for i in range(URW.shape[0]):
    if URW.iloc[i, 3] == "f":
        URWL.append(URW.iloc[i, 1])
#print("Who not Started: ", URWL)

UST = UST[~UST.application_id.isin(URWL)]
print("UST Cleaned Up: ", UST.shape[0])  
UTR = UTR[~UTR.application_id.isin(URWL)]
print("UTR Cleaned Up: ", UTR.shape[0])

UST.to_csv("../data/PreprocessingData/UST_cleaned.csv", sep = ',', index = False)
UTR.to_csv("../data/PreprocessingData/UTR_cleaned.csv", sep = ',', index = False)

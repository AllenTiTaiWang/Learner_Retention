import pandas as pd
import os
cwd = os.getcwd()
import glob
import numpy as np

UST = pd.read_csv(os.path.join(cwd, "../data/PreprocessingData/Ucom.csv"))

#Replace NaN with dummy first, and then we can keep it
UST.update(UST.fillna('dummy'))


UST = UST.join(UST.groupby('user_id')["response"].sum(), on = "user_id", rsuffix = '_tmp')
UST = UST.join(UST.groupby("user_id")["hours_online"].sum(), on = 'user_id', rsuffix = '_tmp')

UST["response"] = UST["response_tmp"]
UST["hours_online"] = UST["hours_online_tmp"]
UST = UST.drop(columns = ['response_tmp', 'hours_online_tmp'])

print(UST)

cols = list(UST)
print(cols)

UST_pivot = UST.pivot_table(index = cols[:len(cols)-2], 
                            columns = 'curriculum_id', values = 'check')


UST_pivot.to_csv("../data/PreprocessingData/UPivot.csv", sep = ",")

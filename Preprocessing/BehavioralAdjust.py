import pandas as pd
import numpy as np
import os
cwd = os.getcwd()
import glob

UTotal = pd.read_csv(os.path.join(cwd, "../data/PreprocessingData/UPivot.csv"))
#UTotal["final_exam"] = UTotal["Final Exam"]
#UTotal.drop(columns = "Final Exam")
UTotal = UTotal.replace('dummy', np.nan)

UTotal.loc[UTotal['Orientation'].notna(), 'Orientation'] = "Attend"
UTotal.Orientation = UTotal.Orientation.fillna("Absence")

UTotal["Orientation"] = np.where((UTotal["program_name"] == "IHeLp 2014 Summer") | (UTotal["program_name"] == "IHeLp 2014 Winter"), 
                                "NoNeed", UTotal["Orientation"])

#UTotal.drop(columns = ['Final Exam'])

Probation = pd.read_csv(os.path.join(cwd, "../data/RawData/probation.csv"))

mapping = dict(Probation[["user_id", "preprobation"]].values)
UTotal["preprobation"] = UTotal.user_id.map(mapping)
UTotal.preprobation = UTotal.preprobation.fillna(0)

mapping = dict(Probation[["user_id", "currentafterpreprbation"]].values)
UTotal["currentafterpreprbation"] = UTotal.user_id.map(mapping)
UTotal.currentafterpreprbation = UTotal.currentafterpreprbation.fillna(0)

UTotal.to_csv("../data/PreprocessingData/UTotal.csv", index = False)

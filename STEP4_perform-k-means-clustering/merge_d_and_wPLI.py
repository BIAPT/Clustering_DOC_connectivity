import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

wPLI = pd.read_csv("../STEP4_perform-k-means-clustering/mode_wpli_Pkmc_K_6_P_7.txt")
dPLI = pd.read_csv("../STEP4_perform-k-means-clustering/mode_dpli_Pkmc_K_6_P_7.txt")

if (wPLI.iloc[1]==dPLI.iloc[1]).all():
    print('ID identic')

P_wPLI = np.array(wPLI.iloc[0][1:])
P_dPLI = np.array(dPLI.iloc[0][1:])

cf_matrix = confusion_matrix(P_wPLI,P_dPLI)

plt.figure()
sns.heatmap(cf_matrix,annot=True,fmt="d")
plt.xlabel("dPLI")
plt.ylabel("wPLI")
plt.savefig("wPLI_dPLI_confusion.jpg")

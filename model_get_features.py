import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pickle

model = pickle.load(open("models/modelv2.sav", "rb"))

feature_list = ["elo", "score", "fgp", "fga3", "fga", "3pp", "ftp", "or", "dr", "ast", "to", "stl", "blk", "pf"]
y = np.arange(14)

importances = model.feature_importances_ 
feature_importance = []

for i in range(len(importances)):
	feature_importance.append(importances[i])
	
plt.figure()
plt.title("Feature Importance") 
plt.bar(y, feature_importance, align="center") 
plt.xticks(y, feature_list) 
plt.xlim([-1, 15])
plt.ylim(0,.6)
plt.xlabel('Feature')
plt.ylabel('Weight')
plt.show()

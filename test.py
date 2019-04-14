# import pandas as pd
# import seaborn
# import matplotlib.pyplot as plt

# df = pd.read_csv("./AAPL.csv")
# data = df.iloc[:, 1:]
# ax = seaborn.heatmap(data.corr())
# plt.show()

from sklearn import tree
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
pre = clf.predict([[1, 1]])
print(pre)
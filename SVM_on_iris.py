import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC

# load iris
iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
df = df.loc[:, 0:1]

# scaling data
scaler = MinMaxScaler()
training = scaler.fit_transform(df.loc[:, 0:1])

# split data
X_train, X_rem, y_train, y_rem = train_test_split(training, iris.target, train_size=0.5, shuffle=True)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.3)

param_grid = {'C': [pow(10, -3), pow(10, -2), pow(10, -1), pow(10, 0), pow(10, 1), pow(10, 2), pow(10, 3)]}
# 10**2 more or less 65 (seen on the lineplot acc-C

# train model and Hyperparametric tuning
# clf = LinearSVC(C=65, random_state=0, tol=1e-6)
clf = SVC(kernel='rbf', C=1)
# clf = GridSearchCV(svc, param_grid)
clf.fit(X_train, y_train)
print(f"Validation set: {clf.score(X_valid, y_valid)}")
# print(clf.best_params_)
print(f"Test set:{clf.score(X_test, y_test)}")

"""
ymin, ymax = -1, 2
w = clf.coef0[0]
a = -w[0] / w[1]
xx = np.linspace(ymin, ymax)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plotting the decision boundary
plt.figure(figsize=(6, 6))
ax = plt.axes()
ax.scatter(X_valid[:, 0], X_valid[:, 1], c=y_valid)
plt.plot(xx, yy, 'k-')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()
"""
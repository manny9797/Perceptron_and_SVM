import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# import some data to play with
iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
df['target'] = iris.target
irisFiltered = df[df['target'] < 2]
X = iris.data
y = iris.target
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.zaxis.set_ticklabels([])

# plt.show()

# SPLITTING DATA INTO £ SETS (TRAINING, VALIDATION, TEST)
scaler = MinMaxScaler()
training = scaler.fit_transform(df.loc[:, 0:1])
X_train, X_rem, y_train, y_rem = train_test_split(training, df['target'], train_size=0.5, shuffle=True)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.3, shuffle=True)

# TRAIN AND RUN PERCEPTRON

param_grid = {'eta0': [0.02, 0.5, 0.4]}
clf = Perceptron(eta0=0.02)
# clf = GridSearchCV(perceptron, param_grid)
clf.fit(X_train, y_train)
print(f"score: {clf.score(X_valid, y_valid)}")
# print(clf.best_params_)
print(classification_report(y_valid, clf.predict(X_valid)))
print(clf.score(X_test, y_test))

# plotting the results

ymin, ymax = -1, 2
w = clf.coef_[0]
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

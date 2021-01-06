import sklearn
# asdfasdf

from sklearn.datasets import load_iris
# from sklearn importl inear_model L
from sklearn.linear_model import LogisticRegression

# LogisticRegression
import numpy as np

X = np.array([0,1, 2, 2.5, 3, 4, 4.5, 5, 6,7]).reshape([-1, 1])
y = np.array([0,0, 0, 1, 0, 0, 1, 1, 0,1])
# X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.predict(X))

print(clf.predict_proba(X[:2, :]))

clf.score(X, y)

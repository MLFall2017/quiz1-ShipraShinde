import numpy as np

X = np.genfromtxt("dataset.csv", delimiter=",")
# delete the first row
X = np.delete(X, (0), axis=0)
np.var(X[:,0])
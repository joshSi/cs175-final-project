# KNN Regression model to be trained
# with various values of k


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor as KNN


def MSE(Y, Yhat):
    return np.sum((Y-Yhat)**2)

data = np.genfromtxt('final.csv', delimiter = ",")
np.random.shuffle(data)

X = data[1:data.shape[0]-1, 1:7]
Y = data[1:data.shape[0]-1, 7:8]
tr = int(X.shape[0]*0.8)
te = X.shape[0] - tr
Xtr = X[0:tr]
Xte = X[tr:tr+te]
Ytr = Y[0:tr]
Yte = Y[tr:tr+te]

k_values = np.arange(1, 600, 3, dtype = int)
errors = []
for k in k_values:
    knn = KNN(n_neighbors=k)
    knn.fit(Xtr, Ytr)
    Yhat = knn.predict(Xte)
    print("K:", k, "MSE:", MSE(Yte, Yhat))
    errors.append(MSE(Yte, Yhat))

#plt.plot(np.arange(1, 201), errors)
#plt.show()


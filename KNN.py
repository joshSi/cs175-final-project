# KNN Regression model to be trained
# with various values of k


import numpy as np 
import pandas as pd 
import sklearn.neighbors as neighbours


data = np.genfromtxt('final.csv', delimiter = ",")
X = data[1:data.shape[0]-1, 1:7]
Y = data[1:data.shape[0]-1, 7:8]
tr = int(X.shape[0]*0.8)
te = X.shape[0] - tr
Xtr = X[0:tr]
Xte = X[tr:tr+te]
Ytr = Y[0:tr]
Yte = Y[tr:tr+te]



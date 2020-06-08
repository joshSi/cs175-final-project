# KNN Regression model to be trained
# with various values of k


import numpy as np 
import pandas as pd 
#import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor as KNN


def shuffle_rows(arr,rows):
    np.random.shuffle(arr[rows[0]:rows[1]+1])

def MSE(Y, Yhat):
    return np.sum((Y-Yhat)**2)

data = np.genfromtxt('final.csv', delimiter = ",")

X = data[1:data.shape[0]-1, 1:7]
Y = data[1:data.shape[0]-1, 7:8]
tr = int(X.shape[0]*0.8)
te = X.shape[0] - tr
Xtr = X[0:tr]
Xte = X[tr:tr+te]
Ytr = Y[0:tr]
Yte = Y[tr:tr+te]

k_values = np.arange(1, 99, 3, dtype = int)
errors_te = []
errors_tr = []
for k in k_values:
    knn = KNN(n_neighbors=k)
    knn.fit(Xtr, Ytr)
    Yhat_te = knn.predict(Xte)
    Yhat_tr = knn.predict(Xtr)
    # print("K:", k, "MSE:", MSE(Yte, Yhat_te))
    errors_te.append(MSE(Yte, Yhat_te))
    errors_tr.append(MSE(Ytr, Yhat_tr))

# plt.title("Errors with varying K values")
# plt.plot(np.arange(1, 34), errors_te, color = "red", label= "Testing; Error")
# plt.plot(np.arange(1, 34), errors_tr, color = "green", label="Training Error")
# plt.xlabel("K values")
# plt.ylabel("MSE")
# plt.legend(loc="upper left")
# plt.show()

def trainAndPredictCases(toPredict):
    nodes_num=[]
    for i in range(3):
        nodes_num.append(4)
    params = nodes_num
    knnreg = KNN(n_neighbors=5)
    knnreg.fit(Xtr, Ytr.ravel())
    
    return knnreg.predict(toPredict)
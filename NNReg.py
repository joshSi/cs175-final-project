# Neural Network with varying parameters

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import VarianceThreshold
import mltools as ml

def MSE(Y, Yhat):
    return np.sum((Y-Yhat)**2)/Y.shape[0]


data = np.genfromtxt('final.csv', delimiter = ",")
#data = np.delete(data, 6, 1)
#data = np.delete(data, 5, 1)
#data = np.delete(data, 4, 1)

X = data[1:data.shape[0]-1, 1:4]
Y = data[1:data.shape[0]-1, 4:5]
print(X)
print(Y)
tr = int(X.shape[0]*0.8)
te = X.shape[0] - tr
Xtr = X[0:tr]
Xte = X[tr:tr+te]
Ytr = Y[0:tr]
Yte = Y[tr:tr+te]

print(Xtr.shape, Ytr.shape, Xte.shape, Yte.shape)


hidden_layers = range(1, 5, 1)
nodes_nums = range(1, 5, 1)

errors_tr = np.zeros((len(hidden_layers),len(nodes_nums)))
errors_te = np.zeros((len(hidden_layers),len(nodes_nums)))

for i in range(len(hidden_layers)):
    hidden_layer = hidden_layers[i]
    for j in range(len(nodes_nums)):
        nodes_num=[]
        for a in range(i):
            nodes_num.append(nodes_nums[j])
        params = nodes_num
        

        print("params:", params)

        nnregr = MLPRegressor(hidden_layer_sizes=params,  random_state=1, max_iter=500, tol = 1e-2)
        nnregr.fit(Xtr, Ytr)
        Yhat_te = nnregr.predict(Xte)
        Yhat_tr = nnregr.predict(Xtr)
        print(Yhat_tr-Ytr)
        errors_tr[i,j] = MSE(Yhat_tr, Ytr)
        errors_te[i,j] = MSE(Yhat_te, Yte)


# Plot the training AUC heatmap
f, ax = plt.subplots(1, 1, figsize=(8, 5))
cax = ax.matshow(errors_tr, interpolation="nearest")
f.colorbar(cax)
ax.set_xticklabels([0]+list(hidden_layers))
ax.set_yticklabels([0]+list(nodes_nums))
ax.set_xlabel("Hidden Layers")
ax.set_ylabel("Number of Nodes")
plt.show()

print(errors_te)
print()
print(errors_tr)




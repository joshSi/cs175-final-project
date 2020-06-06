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

#data = np.delete(data, 6, 1)   # Rain
#data = np.delete(data, 5, 1)   # Wind Speed
#data = np.delete(data, 4, 1)   # Temperature
#data = np.delete(data, 3, 1)   # Elevation
#data = np.delete(data, 2, 1)   # Longitude
#data = np.delete(data, 1, 1)   # Latitude

X = data[1:data.shape[0]-1, 1:4]
Y = data[1:data.shape[0]-1, 4:5]
print(X)
print(Y)
tr = int(X.shape[0]*0.8)
te = X.shape[0] - tr
Xtr = X[0:tr]
Xte = X[tr:tr+te]
Ytr = Y[0:tr]*100
Yte = Y[tr:tr+te]*100


Xtr, params  = ml.rescale(Xtr)
Xte, _  = ml.rescale(Xte, params)

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
        
        #print("params:", params)

        nnregr = MLPRegressor(hidden_layer_sizes=params,  random_state=1, max_iter=1000, 
                             tol = 1e-20, solver = "sgd", activation="identity",
                             learning_rate="adaptive", learning_rate_init=0.001, alpha = 2.5,
                             warm_start = True)
        nnregr.fit(Xtr, Ytr.ravel())
        Yhat_te = nnregr.predict(Xte)
        Yhat_tr = nnregr.predict(Xtr)
        #print(Yhat_tr-Ytr)
        errors_tr[i,j] = MSE(Yhat_tr, Ytr.ravel())
        errors_te[i,j] = MSE(Yhat_te, Yte.ravel())
        print(errors_te[i,j])
        print(errors_tr[i,j])
        print(nnregr.get_params())
        print()

print("errors:")
print("----------------------------")
print(errors_te)
print()
print(errors_tr)
print("----------------------------")


# Plot the training AUC heatmap
f, ax = plt.subplots(1, 2, figsize=(8, 6))
cax = ax[0].matshow(errors_tr, interpolation="nearest")
f.colorbar(cax)
ax[0].set_xticklabels([0]+list(hidden_layers))
ax[0].set_yticklabels([0]+list(nodes_nums))
ax[0].set_xlabel("Hidden Layers")
ax[0].set_ylabel("Number of Nodes")
ax[0].set_title("Training Error")
cax1 = ax[1].matshow(errors_te, interpolation="nearest")
ax[1].set_xticklabels([0]+list(hidden_layers))
ax[1].set_yticklabels([0]+list(nodes_nums))
ax[1].set_title("Testing Error")
ax[1].set_xlabel("Hidden Layers")
ax[1].set_ylabel("Number of Nodes")
plt.show()






# Neural Network with varying parameters

import numpy as np 
import pandas as pd 
# import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import VarianceThreshold
from mltools import rescale
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def MSE(Ytrue, Yhat):
    return np.sum((Ytrue-Yhat)**2)/Ytrue.shape[0]


# Preparing data 
data = np.genfromtxt('final.csv', delimiter = ",")

#data = np.delete(data, 6, 1)   # Rain
#data = np.delete(data, 5, 1)   # Wind Speed
#data = np.delete(data, 4, 1)   # Temperature
#data = np.delete(data, 3, 1)   # Elevation
#data = np.delete(data, 2, 1)   # Longitude
#data = np.delete(data, 1, 1)   # Latitude

X = data[1:data.shape[0]-1, 1:data.shape[1]-1]
Y = data[1:data.shape[0]-1, data.shape[1]-1:data.shape[1]]/100
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state = 1)
Xtr, params  = rescale(Xtr)
Xte, _  = rescale(Xte, params)

# Evaluating performance of models
'''
hidden_layers = range(2, 8, 1)
nodes_nums = range(2, 8, 1)

errors_tr = np.zeros((len(hidden_layers),len(nodes_nums)))
errors_te = np.zeros((len(hidden_layers),len(nodes_nums)))

for i in range(len(hidden_layers)):
    hidden_layer = hidden_layers[i]
    for j in range(len(nodes_nums)):
        nodes_num=[]
        for a in range(i):
            nodes_num.append(nodes_nums[j])
        params = nodes_num
        nnregr = MLPRegressor(hidden_layer_sizes=params,activation = "tanh", alpha = 0.2, learning_rate="invscaling", max_iter=300, 
                 learning_rate_init=0.07,  warm_start=True, early_stopping=True)
        nnregr.fit(Xtr, Ytr.ravel())
        Yhat_te = nnregr.predict(Xte)
        Yhat_tr = nnregr.predict(Xtr)
        try :
            errors_tr[i,j] = round(mean_squared_error(Ytr.ravel(),Yhat_tr),4)
            errors_te[i,j] = round(mean_squared_error(Yte.ravel(),Yhat_te), 4)
            print(errors_te[i,j])
            print(errors_tr[i,j])
            print(nnregr.get_params())
            print()
        except ValueError:
            continue
'''

# Plot the training AUC heatmap
'''
f, ax = plt.subplots(1, 2, figsize=(8, 6))
f.suptitle("Errors of neural networks with varying hidden layers and node numbers \n for Temperature, Rain and Latitude", fontsize=12)
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
'''

def traingAndPredictCases(toPredict):
    nodes_num=[]
    for i in range(3):
        nodes_num.append(4)
    params = nodes_num
    nnregr = MLPRegressor(hidden_layer_sizes=params,activation = "tanh", alpha = 0.2, learning_rate="invscaling", max_iter=300, 
                 learning_rate_init=0.07,  warm_start=True, early_stopping=True)
    nnregr.fit(Xtr, Ytr.ravel())
    
    return nnregr.predict(toPredict)


# Polynomial linear regression model with various degrees
# and features

import numpy as np 
import pandas as pd 
# import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as PolyFeat
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split


def MSE(Y, Yhat):
    return np.sum((Y-Yhat)**2)/Y.shape[0]

# Preparing training and testin data
data = np.genfromtxt('final.csv', delimiter = ",")

# Deleting some of the features

#data = np.delete(data, 6, 1)   # Rain
#data = np.delete(data, 5, 1)   # Wind Speed
#data = np.delete(data, 4, 1)   # Temperature
#data = np.delete(data, 3, 1)   # Elevation
#data = np.delete(data, 2, 1)   # Longitude
#data = np.delete(data, 1, 1)   # Latitude

X = data[1:data.shape[0]-1, 1:data.shape[1]-1]
Y = data[1:data.shape[0]-1, data.shape[1]-1:data.shape[1]]/100
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state = 1)

def getErrors(Xtr, Ytr, Xte, Yte, degrees, mono):
    # Given training and testing sets, gives errors on degrees of 1 to 3
    # includes cross products
    errors_te = []
    errors_tr = []


    for d in degrees:
        if mono:
            poly = PolyFeat(degree = d, interaction_only=True)
        else:
            poly = PolyFeat(degree = d, interaction_only=False)
        XtrP = poly.fit_transform(Xtr)
        XteP = poly.fit_transform(Xte)
        print("Degree:", d)
        print("XtrP XteP shapes:")
        print(XtrP.shape, XteP.shape)
        print(Xtr[0], XtrP[0])
        lin = LinearRegression().fit(XtrP, Ytr)
        Yhat_te = lin.predict(XteP)
        Yhat_tr = lin.predict(XtrP)
        large_pred = Yhat_tr-Ytr
        large_pred = np.argwhere(large_pred>0.1)
        #print("Large pred:", large_pred)
        #print(large_pred.shape)
        
        #print(int(np.sum(abs(Yhat_tr-Ytr))**2)/Ytr.shape[0])
        #print(int(np.sum(abs(Yhat_te-Yte))**2)/Yte.shape[0])
        errors_te.append(mse(Yte, Yhat_te))
        errors_tr.append(mse(Ytr, Yhat_tr))
    return (errors_tr, errors_te)

# Training and plotting MSE
'''
degrees = np.arange(1,4)
err_cross_tr, err_cross_te = getErrors(Xtr, Ytr, Xte, Yte, degrees, False)
err_tr, err_te             = getErrors(Xtr, Ytr, Xte, Yte, degrees, True)

plt.title("Different errors of polynomial regressions with varying degrees \n for Temperature, Rain and Wind Speed")
plt.plot(degrees, err_cross_tr, label = "Training w/ cross prod", color = "blue")
plt.plot(degrees, err_cross_te, label = "Testing w/ cross prod", color = "#053853")
plt.plot(degrees, err_tr, label = "Training w/o cross prod", color = "red")
plt.plot(degrees, err_te, label = "Testing w/o cross prod", color = "#A50000")
plt.xlabel("Degrees")
plt.ylabel("MSE")
plt.legend(loc="upper left")
plt.show()
'''






def trainAndPredictCases(toPred):
    poly = PolyFeat(degree = 3, interaction_only=True)
    XtrP = poly.fit_transform(Xtr)
    lin = LinearRegression().fit(XtrP, Ytr)
    toPred = poly.fit_transform(toPred)
    return lin.predict(toPred)
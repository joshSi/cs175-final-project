# Polynomial linear regression model with various degrees

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as PolyFeat
from sklearn.linear_model import LinearRegression as LinReg
import mltools as ml
from sklearn.linear_model import LassoCV


def MSE(Y, Yhat):
    return np.sum((Y-Yhat)**2)/Y.shape[0]

data = np.genfromtxt('final.csv', delimiter = ",")
features = ["Latitude", "Longitude", "Elevation", "Temperature", "WindSpeed", "Rain"]
rows = [1, data.shape[0]-1]
np.random.shuffle(data[rows[0]:rows[1]+1])

X = data[1:data.shape[0]-1, 1:7]
Y = data[1:data.shape[0]-1, 7:8]
tr = int(X.shape[0]*0.8)
te = X.shape[0] - tr
Xtr = X[0:tr]
Xte = X[tr:tr+te]
Ytr = Y[0:tr]
Yte = Y[tr:tr+te]

# Using LassoCV to find coef of important features
lcv = LassoCV().fit(Xtr, Ytr)
importance = np.abs(lcv.coef_)
coef = {}
for i in range(6):
    coef[features[i]] = importance[i]

coef1 = sorted(coef.items(), key=lambda x: x[1], reverse=True)
for (f,c) in coef1:
    print("Feature:", f, "\nCoef:", c)

'''
Feature: Latitude 
Coef: 0.02152277460760391
Feature: Temperature 
Coef: 0.01048501111894712
Feature: Rain 
Coef: 0.0011337726939069585
Feature: Longitude 
Coef: 0.0007183706504282805
Feature: WindSpeed 
Coef: 2.1580535226731523e-05
Feature: Elevation 
Coef: 1.2414793809479373e-05
'''

# Using Pandas correlation
data1 = pd.read_csv("final.csv", sep = ",")
cor = data1.corr()
print(cor)

'''
                Latitude    Longitude  Elevation  Temperature  Windspeed   Rain     Growth %

Latitude        1.000000     0.281789   0.091905    -0.756511  -0.001808 -0.057583 -0.002971
Longitude       0.281789     1.000000  -0.333725    -0.266655  -0.040929 -0.120775  0.053067
Elevation       0.091905    -0.333725   1.000000    -0.119441   0.144285  0.048483 -0.009103
Temperature    -0.756511    -0.266655  -0.119441     1.000000  -0.015703  0.001107 -0.177541
Windspeed      -0.001808    -0.040929   0.144285    -0.015703   1.000000 -0.023887 -0.017797
Rain           -0.057583    -0.120775   0.048483     0.001107  -0.023887  1.000000  0.051253
Growth %       -0.002971     0.053067  -0.009103    -0.177541  -0.017797  0.051253  1.000000
'''

'''
degrees = np.arange(1, 5, 1, dtype = int)
print(degrees)
errors_te = []
errors_tr = []
for d in degrees:
    poly1 = PolyFeat(degree = d, interaction_only = False)
    poly2 = PolyFeat(degree = d, interaction_only = True)
    Xtr = poly2.fit_transform(Xtr)
    Xte = poly2.fit_transform(Xte)
    lin = LinReg()
    lin.fit(Xtr, Ytr)

    Yhat_te = lin.predict(Xte)
    Yhat_tr = lin.predict(Xtr)
    print("Degree:", d, "MSE:", MSE(Yte, Yhat_te))
    errors_te.append(MSE(Yte, Yhat_te))
    errors_tr.append(MSE(Ytr, Yhat_tr))
    '''

'''
    Xtr = ml.transforms.fpoly_mono(Xtr, d)
    Xte = ml.transforms.fpoly_mono(Xte, d)
    lin = ml.linear.linearRegress()
    lin.train(Xtr, Ytr)
    Yhat_te = lin.predict(Xte)
    Yhat_tr = lin.predict(Xtr)
    errors_te.append(MSE(Yte, Yhat_te))
    errors_tr.append(MSE(Ytr, Yhat_tr))
    '''
'''
plt.title("Errors with products of N degrees, only cross products")
plt.plot(np.arange(1, 7), errors_te, color = "red", label= "Testing; Error")
plt.plot(np.arange(1, 7), errors_tr, color = "green", label="Training Error")
plt.xlabel("Degrees")
plt.ylabel("MSE")
plt.legend(loc="upper left")
plt.show()
'''

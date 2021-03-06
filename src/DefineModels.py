import numpy as np
from Regressor import Regressor
from HelpFunctions import *

from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor, Ridge, Lasso, RidgeCV, LassoCV, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

random_state = 123

## SET PARAMETERS
# Params KNN Regressor
n_neighbors_temp = [2, 4, 8, 12]
weights_temp = ['uniform', 'distance']
params_knn = [{'n_neighbors': i, 'weights': j} for i in n_neighbors_temp for j in weights_temp]

# Params Huber
params_huber = [{'epsilon':i} for i in np.linspace(1, 2, 6)]

# Params Ridge
params_ridge = [{'alpha':i} for i in [.1, .9, 1, 2, 4]]

# Params Lasso
params_lasso = [{'alpha':i} for i in [.1, .9, 1, 2, 4]]

# Params Decistion Tree
max_depth_temp = [(i+1)*3 for i in range(5)]
min_samples_split_temp = [(i+1)*3 for i in range(3)]
max_features_temp = ["sqrt", "log2", "auto"]
params_dt = [{'max_depth':i, 'min_samples_split':j, 'max_features':k} for i in max_depth_temp for j in min_samples_split_temp for k in max_features_temp]

# Params Random Forest
n_estimators_temp = [10, 50, 200]
max_depth_temp = [10, 50]
min_samples_split_temp = [2,5]
params_rf = [{"n_estimators": i, 'max_depth': j, 'min_samples_split': k} for i in n_estimators_temp for j in max_depth_temp for k in min_samples_split_temp]

# Params XGBOOST
eta_temp = np.linspace(0.01, 0.3, 2) #10
max_depth_temp = np.linspace(3, 8, 2, dtype= np.dtype(np.int16)) #8
gamma_temp = np.linspace(0, 0.2, 2) #10
subsample_temp = np.linspace(0.5, 1, 2) #6
colsample_bytree_temp = np.linspace(0.5, 1, 2) #6
alpha_temp = np.linspace(0, 0.1, 2) #11
min_child_weight_temp = np.linspace(1, 20, 2) #10

params_xgb = [{'objective':  'reg:squarederror','eta':i, 'max_depth':j, 'gamma':k, 'subsample':l, 'colsample_bytree':m, 'alpha':n, 'min_child_weight':o} for i in eta_temp for j in max_depth_temp for k in gamma_temp for l in subsample_temp for m in colsample_bytree_temp for n in alpha_temp for o in min_child_weight_temp]
models = []

# ## ADD MODELS
# # add linear models
models += [Regressor("OLS", LinearRegression, [{}])]
models += [Regressor("ThSen", TheilSenRegressor, [{}])] # (very slow)
models += [Regressor("Huber", HuberRegressor, params_huber)]

# # ridge and lasso
models += [Regressor("Ridge", Ridge, params_ridge)]
models += [Regressor("Lasso", Lasso, params_lasso)]
# # Bayesian Ridge
models += [Regressor("BayRidge", BayesianRidge, [{}])]
#
# KNN Regressor
models += [Regressor("KNN", KNeighborsRegressor, params_knn)]

# add tree
models += [Regressor("DecTree", DecisionTreeRegressor, params_dt)]

# forest
models += [Regressor("Forest", RandomForestRegressor, params_rf)]

# xgboost
models += [Regressor("XGBoost", xgb.XGBRegressor, params_xgb)]

print("Models loaded succesfully on variable 'models'")

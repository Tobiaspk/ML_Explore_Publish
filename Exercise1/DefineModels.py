import numpy as np
from DataSetting import *
from Regressor import * 
from HelpFunctions import *

from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.linear_model import RidgeCV, LassoCV, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

random_state = 123

## SET PARAMETERS
# Params KNN Regressor
n_neighbors_temp = [(3*i) for i in range(1,6)]
weights_temp = ['uniform', 'distance']
algorithm_temp = ['ball_tree', 'kd_tree', 'brute', 'auto']
params_knn = [{'n_neighbors': i, 'weights': j, 'algorithm': k} for i in n_neighbors_temp for j in weights_temp for k in algorithm_temp]

# Params Ridge
params_ridge = [{"cv":5}]

# Params Lasso
params_lasso = [{"cv":5}]

# Params Decistion Tree
max_depth_temp = [(i+1)*3 for i in range(5)]
max_leaf_nodes_temp = [None] + [(i+1)*3 for i in range(4)]
min_samples_splits_temp = np.linspace(0.1, 1.0, 5)
params_dt = [{'max_depth':i, 'max_leaf_nodes':j, 'min_samples_split':k} for i in max_depth_temp for j in max_leaf_nodes_temp for k in min_samples_splits_temp]

# Params Random Forest
n_estimators_temp = np.linspace(100, 1000, 2, dtype= np.dtype(np.int16))  #10
max_depth_temp = np.linspace(10, 100, 2) #10
min_samples_split_temp = [2,5]# [2, 5, 8, 11]
min_samples_leaf_temp = [1]# [1, 3, 5]
#params_rf = [{"n_estimators":i, 'max_depth':j, 'min_samples_split':k, 'min_samples_leaf':l} for i in n_estimators_temp for j in max_depth_temp for k in min_samples_split_temp for l in min_samples_leaf_temp]
params_rf = [{"n_estimators": 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf':1}]

# Params XGBOOST
eta_temp = np.linspace(0.01, 0.3, 5) #10
max_depth_temp = np.linspace(3, 8, 4, dtype= np.dtype(np.int16)) #8
gamma_temp = np.linspace(0, 0.2, 5) #10
subsample_temp = np.linspace(0.5, 1, 3) #6
colsample_bytree_temp = np.linspace(0.5, 1, 3) #6
alpha_temp = np.linspace(0, 0.1, 5) #11
min_child_weight_temp = np.linspace(1, 20, 5) #10

params_xgb = [{'objective':  'reg:squarederror','eta':i, 'max_depth':j, 'gamma':k, 'subsample':l, 'colsample_bytree':m, 'alpha':n, 'min_child_weight':o} for i in eta_temp for j in max_depth_temp for k in gamma_temp for l in subsample_temp for m in colsample_bytree_temp for n in alpha_temp for o in min_child_weight_temp]
#params_xgb = [{'eta':0.02, 'max_depth':3, 'gamma':0, 'subsample':0.5, 'colsample_bytree':0.5, 'alpha':0, 'min_child_weight': 1}]
models = []

## ADD MODELS
# add linear models
models += [Regressor("OLS", LinearRegression, [{}])]
models += [Regressor("ThSen", TheilSenRegressor, [{'random_state':random_state}])]
#models += [Regressor("Ransac", RANSACRegressor, [{'random_state':random_state}])]
models += [Regressor("Huber", HuberRegressor, [{}])]

# ridge and lasso
#models += [Regressor("Ridge", RidgeCV, params_ridge)]
models += [Regressor("Lasso", LassoCV, params_lasso)]

# Bayesian Ridge
models += [Regressor("BayRidge", BayesianRidge, [{}])]

# KNN Regressor
#models += [Regressor("KNN", KNeighborsRegressor, params_knn)]

# add tree
#models += [Regressor("DecTree", DecisionTreeRegressor, params_dt)]

# forest
#models += [Regressor("Forest", RandomForestRegressor, params_rf)]

# xgboost
#models += [Regressor("XGBoost", xgb.XGBRegressor, params_xgb)]
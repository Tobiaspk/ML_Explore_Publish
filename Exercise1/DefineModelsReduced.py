import numpy as np
from DataSetting import *
from Regressor import * 
from HelpFunctions import *

from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor, Ridge, Lasso
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
params_ridge = [{'alpha':i} for i in [2, 4, 8, 12]]

# Params Lasso
params_lasso = [{'alpha':i} for i in [2, 4, 8, 12]]

# Params Decistion Tree
max_depth_temp = [(i+1)*3 for i in range(5)]
min_samples_split_temp = [(i+1)*3 for i in range(3)]
max_features_temp = ["sqrt", "log2", "auto"]
params_dt = [{'max_depth':i, 'min_samples_split':j, 'max_features':k} for i in max_depth_temp for j in min_samples_split_temp for k in max_features_temp]

# Params Random Forest
n_estimators_temp = [10, 50]
max_depth_temp = [5, 15]
min_samples_split_temp = [5]
params_rf = [{"n_estimators": i, 'max_depth': j, 'min_samples_split': k} for i in n_estimators_temp for j in max_depth_temp for k in min_samples_split_temp]

# Params XGBOOST
eta_temp = [.1, .3]
gamma_temp = [.2, 1]
lambda_temp = [1.5, 3]

params_xgb = [{"eta": i, "gamma": j, "lambda": k} for i in eta_temp for j in gamma_temp for k in lambda_temp]


########### MODELS
models_reduced = []

## ADD MODELS
# add linear models
models_reduced += [Regressor("OLS", LinearRegression, [{}])]
models_reduced += [Regressor("Huber", HuberRegressor, params_huber)]

# ridge and lasso
models_reduced += [Regressor("Ridge", Ridge, params_ridge)]
models_reduced += [Regressor("Lasso", Lasso, params_lasso)]

# KNN Regressor
models_reduced += [Regressor("KNN", KNeighborsRegressor, params_knn)]

# add tree
models_reduced += [Regressor("DecTree", DecisionTreeRegressor, params_dt)]

# forest
models_reduced += [Regressor("Forest", RandomForestRegressor, params_rf)]

# xgboost
models_reduced += [Regressor("XGBoost", xgb.XGBRegressor, params_xgb)]


print("Reduced Models loaded succesfully on variable 'models_reduced'")
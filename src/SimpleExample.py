# import numpy
import numpy as np

# load classes
from DataSetting import *
from Regressor import *
from HelpFunctions import *

# choose two models
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# get a dataset
from sklearn.datasets import load_boston
dat = load_boston()
x = dat['data']
y = dat['target']

# set parameters for models as dict
parameters_randomforest = [{'max_depth':3, 'min_samples_leaf':1},
                           {'max_depth':6, 'bootstrap':False}]

parameters_decisiontree = [{'max_depth':1, 'max_leaf_nodes':2},
                           {'max_depth':3, 'max_leaf_nodes':2},
                           {'max_depth':5, 'max_leaf_nodes':5},
                           {'max_depth':7, 'max_leaf_nodes':5}]

# create Regressors (a Regressor is one algorithm and a list of parameters)
models = [Regressor(name="RandomForest",
                    model=RandomForestRegressor,
                    parameters=parameters_randomforest),
          Regressor(name="DecisionTree",
                    model=DecisionTreeRegressor,
                    parameters=parameters_decisiontree)]

# create a DataSetting
ds = DataSetting(y=y, x=x, models=models, loss_function=rmse)

# fit all models
ds.evaluate_all()

# show results
ds.collect_losses()
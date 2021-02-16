import pandas as pd
from DataSetting import DataSetting
from HelpFunctions import *
from DefineModels import models


# define simple loss function used in this practise
def rmse(y, prediction):
    return np.mean(np.square(y - prediction))**.5


# Follow the steps
# 1. Load Data
# (1b). Define Regressors and Parameter Grids
# 2. Define Datasettings
# 3. Run evaluate_all()
# 4. Plot validation or losses

## Read Data
AutoTrain = pd.read_csv("datasets/AutoTrain.csv", na_values="?")
BikeTrain = pd.read_csv("datasets/BikeTrain.csv")

Facebook = pd.read_csv("datasets/Facebook.csv", header = None)
RealEstate = pd.read_csv("datasets/RealEstate.csv")


## EVALUATE AUTO TRAIN
AutoTrain.horsepower = AutoTrain.horsepower.fillna(AutoTrain.horsepower.mean())
ds_AutoTrain = DataSetting(y=AutoTrain["mpg"],
                            x=AutoTrain.drop(["id", "mpg", "carName"], 1),
                           models=models.copy(),
                           loss_function=rmse,
                           k=5)
ds_AutoTrain.evaluate_all()
ds_AutoTrain.plot_model_validation_curves(path='plots/Auto/AutoLearningCurves.png')


## EVALUATE BIKETRAIN
ds_BikeTrain = DataSetting(y= BikeTrain["cnt"],
                           x = BikeTrain.drop(["id", "cnt", "dteday"], 1),
                           models = models.copy(),
                           loss_function = rmse,
                           k = 5)
ds_BikeTrain.evaluate_all()
ds_BikeTrain.plot_model_learning_curve(path='plots/Bike/BikeLearningCurves')

## EVALUEATE FACEBOOK

ds_FaceBook = DataSetting(y = Facebook.iloc[:,53],
                          x = Facebook.drop(columns = [53]),
                          models = models.copy(),
                          loss_function= rmse,
                          k=5)
ds_FaceBook.normalize_data()
ds_FaceBook.evaluate_all()
ds_FaceBook.plot_model_learning_curve(path='plots/FaceBook/FaceBookLEaningCurves')


## EVALUATE REALESTATE

ds_RealEstate = DataSetting(y = RealEstate["Y house price of unit area"],
                            x = RealEstate.drop(columns = ["Y house price of unit area"]),
                            models = models.copy(),
                            loss_function= rmse,
                            k = 5)
ds_RealEstate.evaluate_all()
ds_RealEstate.plot_model_learning_curve(path='plots/RealEstate/RealEstateLearningCurves')

import numpy as np
import pandas as pd
from DataSetting import *
from Regressor import * 
from HelpFunctions import *
#from DefineModels import *
from DefineModelsReduced import *



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
#print("#"*20 + "   AUTO TRAIN   " + "#"*20)
#print(ds_AutoTrain.losses_to_pandas_long())


## EVALUATE BIKETRAIN
ds_BikeTrain = DataSetting(y= BikeTrain["cnt"],
                           x = BikeTrain.drop(["id", "cnt", "dteday"], 1),
                           models = models.copy(),
                           loss_function = rmse,
                           k = 5)

print("\n\n")
ds_BikeTrain.evaluate_all()
#print("#"*20 + "   BIKE TRAIN   " + "#"*20)
#print(ds_BikeTrain.losses_to_pandas_long())
ds_BikeTrain.plot_model_learning_curve(path='plots/Bike/BikeLearningCurves')

## EVALUEATE FACEBOOK

ds_FaceBook = DataSetting(y = Facebook.iloc[:,53],
                          x = Facebook.drop(columns = [53]),
                          models = models.copy(),
                          loss_function= rmse,
                          k=5)
#ds_FaceBook.normalize_data()
ds_FaceBook.evaluate_all()
ds_FaceBook.plot_model_learning_curve(path='plots/FaceBook/FaceBookLEaningCurves')
#print("#"*20 + "   FACEBOOK   " + "#"*20)
#print(ds_FaceBook.losses_to_pandas_long())


## EVALUATE REALESTATE

ds_RealEstate = DataSetting(y = RealEstate["Y house price of unit area"],
                            x = RealEstate.drop(columns = ["Y house price of unit area"]),
                            models = models.copy(),
                            loss_function= rmse,
                            k = 5)



ds_RealEstate.evaluate_all()
ds_RealEstate.plot_model_learning_curve(path='plots/RealEstate/RealEstateLearningCurves')
#print("#"*20 + "   REALESTATE   " + "#"*20)
#print(ds_RealEstate.losses_to_pandas_long())





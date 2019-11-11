import numpy as np
from time import time
from sklearn.model_selection import KFold
from scipy import stats
import pandas as pd
class DataSetting:
    def __init__(self, y, x, models, loss_function, k=5):
        self.y = y
        self.x = x
        self.models = models
        self.loss_function = loss_function
        self.k = k
        
        for mod in self.models:
            mod.loss_function = self.loss_function
        
    def generate_cv(self, k):
        kf = KFold(n_splits=k)
        return kf.split(self.x)
    
    def evaluate_all(self):
        for mod in self.models:
            begin = time()
            mod.fit_all(ds=self,k=self.k)
            print(mod.name, "done in", np.round(time() - begin, 3), "seconds.")
            print("\n")
            
    def get_data(self):
        return(self.x, self.y)

    def normalize_data(self):
        normX = (self.x-self.x.mean())/self.x.std() ##if regressor doesnt normalize data
        self = normX;


    def collect_losses(self):
        for mod in self.models:
            for i in range(len(mod.parameters)):
                print(mod.name, "\t", np.round(mod.losses[i], 3), "\t","mean: ",
                      np.round(np.mean(mod.losses[i]),3), "std.dev :" ,np.round(np.std(mod.losses[i]),3), "\t",
                      mod.parameters[i])
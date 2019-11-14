import numpy as np
import pandas as pd
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
        begin_all = time()
        for mod in self.models:
            begin = time()
            mod.fit_all(ds=self,k=self.k)
            print(mod.name, "done in", np.round(time() - begin, 3), " seconds.")
            print("\n")
        print("Everything evaluated in " + str(np.round(time() - begin_all, 3)) + " seconds")
            
    def get_data(self):
        return(self.x, self.y)

    def normalize_data(self):
        normX = (self.x-self.x.mean())/self.x.std() ##if regressor doesnt normalize data
        self = normX;


    def losses_to_pandas(self):
        colnames = ("Algorithm", "Losses", "LossesMean", "LossesSD", "Parameters")
        df = dict(zip(colnames, [[] for i in range(len(colnames))]))
        
        for mod in self.models:
            for i in range(len(mod.parameters)):
                df["Algorithm"].append(mod.name)
                df["Losses"].append(np.round(mod.losses[i], 3))
                df["LossesMean"].append(np.round(np.mean(mod.losses[i]),3))
                df["LossesSD"].append(np.round(np.std(mod.losses[i]),3))
                df["Parameters"].append(mod.parameters[i])
        df = pd.DataFrame(df)
        return(df)
                      
                      
                      
                      
                      
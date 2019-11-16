import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from sklearn.model_selection import KFold
from scipy import stats

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
    
    def evaluate_all(self, verbose=2):
        # 0 ... no outputs, 1 ... output after a each model that is finished, 2 ... all outputs
        begin_all = time()
        for mod in self.models:
            begin = time()
            mod.fit_all(ds=self,k=self.k,verbose=verbose)
            if verbose >= 1: print(mod.name, "done in", np.round(time() - begin, 3), " seconds.")
        print("Everything evaluated in " + str(np.round(time() - begin_all, 3)) + " seconds.\n\n")
            
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
    
    
    def losses_to_pandas_long(self):
        losses = self.losses_to_pandas()
        long = pd.DataFrame({"Algorithm":np.repeat(losses.Algorithm.values, self.k),
                             "Loss":np.concatenate(losses.Losses.values).ravel()})
        return(long)

    def min_losses(self):
        losses = self.losses_to_pandas()
        mins = losses.groupby("Algorithm")["LossesMean"].min()
        mins_out = dict(zip(mins.axes[0], mins.values))
        return(mins_out)
    
    def boxplot_losses(self, ax=None, *args):
        losses_long = self.losses_to_pandas_long()
        sns.boxplot(losses_long.Algorithm, losses_long.Loss, ax=ax, *args)

    def boxplot_losses_min(self, ax=None, *args):
        min_losses_id = self.losses_to_pandas().groupby("Algorithm")["LossesMean"].idxmin().values
        losses = self.losses_to_pandas().iloc[min_losses_id]
        losses_long = pd.DataFrame({"Algorithm":np.repeat(losses.Algorithm.values, self.k),
                             "Loss":np.concatenate(losses.Losses.values).ravel()})
        sns.boxplot(losses_long.Algorithm, losses_long.Loss, ax=ax, *args)
        
    def barplot_losses_min(self, ax=None, *args):
        losses_min = self.min_losses()
        sns.barplot(list(losses_min.keys()), list(losses_min.values()), ax=ax, *args)
                      
                      
                      
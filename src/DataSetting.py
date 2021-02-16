import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from sklearn.model_selection import KFold
from plot_learning_curve import *
from scipy import stats
import matplotlib.patches as mpatches
from sklearn.model_selection import validation_curve


class DataSetting:
    """
    Stores information about the dataset.
    :param y: Numpy array of dependent variable
    :param x: Numpy array of independent variables
    :param models: List of Regressor objects defining the model type and parameter grid
    :param loss_function: A loss function, that takes (y, prediction) as input.
    """
    def __init__(self, y, x, models, loss_function, k=5):
        self.y = y
        self.x = x
        self.models = models
        self.loss_function = loss_function
        self.k = k
        
        for mod in self.models:
            mod.loss_function = self.loss_function
        
    def generate_cv(self, k):
        # generate simple cross validation (many parameters neglected for simplicity)
        kf = KFold(n_splits=k)
        return kf.split(self.x)
    
    def evaluate_all(self, verbose=2):
        # Calls methods fit_all() on each regressor model
        # verbose:
        # - 0 ... no outputs
        # - 1 ... output after a each model that is finished
        # - 2 ... all outputs
        begin_all = time()
        for mod in self.models:
            begin = time()
            mod.fit_all(ds=self,k=self.k,verbose=verbose)
            if verbose >= 1: print(mod.name, "done in", np.round(time() - begin, 3), " seconds.")
        print("Everything evaluated in " + str(np.round(time() - begin_all, 3)) + " seconds.\n\n")
            
    def get_data(self):
        # Return an (dependent, independent) variable tuple
        return self.x, self.y

    def normalize_data(self):
        # Scale all x variables to mean 0, variance 1
        normX = (self.x-self.x.mean())/self.x.std() ##if regressor doesnt normalize data
        self = normX;

    def losses_to_pandas(self):
        # Collect all losses and return them in an exhaustive pandas dataframe
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
        # Transforms all losses into a long format
        losses = self.losses_to_pandas()
        long = pd.DataFrame({"Algorithm":np.repeat(losses.Algorithm.values, self.k),
                             "Loss":np.concatenate(losses.Losses.values).ravel()})
        return(long)

    def min_losses(self):
        # Get minimum loss for each algorithm
        losses = self.losses_to_pandas()
        mins = losses.groupby("Algorithm")["LossesMean"].min()
        mins_out = dict(zip(mins.axes[0], mins.values))
        return(mins_out)
    
    def boxplot_losses(self, ax=None, *args):
        # Display boxplots of the losses of all algorithms
        losses_long = self.losses_to_pandas_long()
        sns.boxplot(losses_long.Algorithm, losses_long.Loss, ax=ax, *args)

    def boxplot_losses_min(self, ax=None, *args):
        # Display boxplots of the losses of the best parameter settings
        min_losses_id = self.losses_to_pandas().groupby("Algorithm")["LossesMean"].idxmin().values
        losses = self.losses_to_pandas().iloc[min_losses_id]
        losses_long = pd.DataFrame({"Algorithm":np.repeat(losses.Algorithm.values, self.k),
                                    "Loss":np.concatenate(losses.Losses.values).ravel()})
        sns.boxplot(losses_long.Algorithm, losses_long.Loss, ax=ax, *args)
        
    def barplot_losses_min2(self, ax=None, *args):
        losses_min = self.min_losses()
        sns.barplot(list(losses_min.keys()), list(losses_min.values()), ax=ax, *args)

    def plot_model_learning_curves(self, path = ''):
        # Plot the learning curves of the models
        k = 0
        size = sum((len(mod.parameters)>1) for mod in self.models)
        fig, axes = plt.subplots(2,size, sharex='col', sharey='row')

        for mod in self.models:
            if(len(mod.parameters) >1):
                tempModel = mod.set_params(mod.parameters[mod.best_params])
                title = mod.name + ": best Parameters"
                plot_learning_curve(tempModel, title, self.x, self.y, axis = axes[0,k])

                tempModel = mod.set_params(mod.parameters[mod.worst_params])
                title = mod.name + ": worst Parameters"
                plot_learning_curve(tempModel, title, self.x, self.y, axis = axes[1,k])

                k += 1

        red_patch = mpatches.Patch(color='red', label='Training score')
        green_patch = mpatches.Patch(color='green', label='Cross-validation score')
        fig.legend(handles=[red_patch,green_patch])
        fig.text(0.5, 0.04, 'Training Examples', ha='center')
        fig.text(0.04, 0.5, 'Score', va='center', rotation='vertical')
        plt.savefig(path)
        plt.show()

    def plot_model_validation_curves(self,path=''): 
        # Plot the validation curve for only one model
        param_range = np.linspace(2,101,20, dtype= np.dtype(np.int16))
        train_scores, test_scores = validation_curve(
            self.models[0].set_params(self.models[0].parameters[self.models[0].best_params]), self.x, self.y,
            param_name="alpha", param_range=param_range,
            cv=5,  n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Validation Curve with Lasso")
        plt.xlabel(r"Alpha")
        plt.ylabel("Score")
        lw = 2
        plt.plot(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        plt.show()







                      
                      
                      
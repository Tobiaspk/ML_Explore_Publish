import numpy as np

class Regressor:
    """
    Wraps a regressor to simplify working in this framework
    :param name: Name of the model
    :param model: Class of the respective model (for example LinearRegression from sklearn.linear_model)
    :param parameters: List of dictionaries of parameters (forexample params=[{'max_depth':5}, {'max_depth':10}])
    :param loss_function: Will be filled depending on datasetting.
    """
    def __init__(self, name, model, parameters, loss_function=None):
        self.name = name
        self.model = model
        self.parameters = parameters
        self.losses = np.zeros(len(parameters))
        self.loss_function = loss_function
        self.best_params = None
        self.worst_params = None

    def fit_all(self, ds, k, verbose):
        # For each parameter setting fit and evaluate fit using k-fold cross validation
        # param ds: DataSetting instance
        # param k: k-fold cross validation
        # verbose: print outputs after each iteration if >= 2
        x,y = ds.get_data()
        self.losses = np.zeros((len(self.parameters), k))
        
        cvlosses = np.zeros(k);
        
        for i in range(len(self.parameters)):
            l = 0;
            for train_index, test_index in ds.generate_cv(k=k):
                cvlosses[l], mod = self.fit_one(y.loc[train_index], 
                                                x.loc[train_index],
                                                y.loc[test_index],
                                                x.loc[test_index],
                                                param_i=i)
                l +=1;
                
            if verbose >= 2: print(self.name + " parameter " + str(i+1) + "/" + str(len(self.parameters)) + " done")
            self.losses[i,:] = cvlosses
                
        # set best parameters
        self.best_params = np.argmin(np.mean(self.losses, 1)) ## calculate average among CVs
        self.worst_params = np.argmax(np.mean(self.losses, 1))
    
    def fit_one(self, y, x, y_test, x_test, param_i):
        # Fit one parameter settings
        # assign parameters
        param = self.parameters[param_i]
        
        # set parameters
        model_temp = self.set_params(param=param)
        
        # fit model
        model_temp.fit(y=y, X=x)
        
        # make prediction
        prediction = model_temp.predict(x_test)
        
        # evaluate and save loss
        loss = self.get_loss(y_test, prediction)
        
        # return loss and model
        return(loss, model_temp)
    
    def set_params(self, param):
        # Set the temporary parameters of the model
        # parameters must be dict
        if (type(param) != dict):
            print("Parameters must be type 'dict'")
            return 0
        
        # set parameter to model
        model_new = self.model(**param)
        return model_new
        
    def get_loss(self, y, prediction):
        # Apply loss function
        return self.loss_function(y, prediction)

import numpy as np

class Regressor:
    def __init__(self, name, model, parameters, loss_function=None):
        self.name = name
        self.model = model
        self.parameters = parameters
        self.losses = np.zeros(len(parameters))
        self.loss_function = loss_function
        self.best_params = None

    def fit_all(self, ds, k):
        x,y = ds.get_data()
        self.losses = np.zeros((len(self.parameters), k))
        
        cvlosses = np.zeros(k);
        
        for i in range(len(self.parameters)):
            print(self.name + " parameter " + str(i+1) + "/" + str(len(self.parameters)) + " done")
            l = 0;
            for train_index, test_index in ds.generate_cv(k=k):
                cvlosses[l], mod = self.fit_one(y.loc[train_index], 
                                                x.loc[train_index],
                                                y.loc[test_index],
                                                x.loc[test_index],
                                                param_i=i)
                l +=1;
                
            self.losses[i,:] = cvlosses
                
        # set best parameters
        self.best_params = np.argmin(np.mean(self.losses, 0))
    
    def fit_one(self, y, x, y_test, x_test, param_i):
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
        # parameters must be dict
        if (type(param) != dict):
            print("Parameters must be type 'dict'")
            return(0)
        
        # set parameter to model
        model_new = self.model(**param)
        return(model_new)
        
    def get_loss(self, y, prediction):
        # apply loss function
        loss = self.loss_function(y, prediction)
        return(loss)
import numpy as np

def rmse(y, prediction):
    loss = np.mean(np.square(y - prediction))**.5
    return(loss)

def test(a):
	print(a)